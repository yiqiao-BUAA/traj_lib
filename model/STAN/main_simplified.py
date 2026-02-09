# model/STAN/main.py
# -*- coding: utf-8 -*-

import math
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.logger import get_logger
from utils.exargs import ConfigResolver

# ---------------------------------------------------------------------------
# Load model hyper-parameters from YAML
# ---------------------------------------------------------------------------
model_args = ConfigResolver("./model/STAN/STAN.yaml").parse()
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Views (not used for this model, keep empty to align with framework)
# ---------------------------------------------------------------------------
pre_views: list[str] = []  # no pre-views
post_views: list[str] = []  # no post-views

# ---------------------------------------------------------------------------
# Global states
# ---------------------------------------------------------------------------
model: Optional[nn.Module] = None
_device: Optional[torch.device] = None


# ============================== Utilities ==================================
def _resize_embedding(
    old_emb: nn.Embedding, new_num: int, init_std: float = 0.02
) -> nn.Embedding:
    """
    Expand an embedding matrix while keeping existing weights.

    Args:
        old_emb: existing embedding to expand
        new_num: new number of embeddings (rows)
        init_std: std for normal init for new rows

    Returns:
        Expanded nn.Embedding with copied old weights and initialized new rows.
    """
    if new_num <= old_emb.num_embeddings:
        return old_emb

    new_emb = nn.Embedding(
        new_num, old_emb.embedding_dim, padding_idx=old_emb.padding_idx
    )
    with torch.no_grad():
        # Initialize all weights, then overwrite the old range to keep them intact.
        nn.init.normal_(new_emb.weight, mean=0.0, std=init_std)
        new_emb.weight[: old_emb.num_embeddings].copy_(old_emb.weight)
        if old_emb.padding_idx is not None:
            new_emb.weight[old_emb.padding_idx].zero_()
    return new_emb


def _maybe_expand_from_batch(batch: Dict[str, Any], model_: nn.Module) -> None:
    """
    Expand embeddings/pos-embedding/out-bias on the fly based on the current batch.

    Inspects maximum IDs and sequence length in the batch and expands if needed.

    Shapes:
        batch["POI_id"] : [B, S]
        batch["user_id"]: [B]
    """
    poi_max: int = int(batch["POI_id"].max().item())
    user_max: int = int(batch["user_id"].max().item())
    S: int = int(batch["POI_id"].size(1))

    need_expand = False
    new_num_poi = model_.poi_emb.num_embeddings
    new_num_user = model_.user_emb.num_embeddings
    new_max_S = model_.pos_emb.num_embeddings

    if poi_max + 1 > new_num_poi:
        new_num_poi = poi_max + 1
        need_expand = True
    if user_max + 1 > new_num_user:
        new_num_user = user_max + 1
        need_expand = True
    if S > new_max_S:
        new_max_S = S
        need_expand = True

    if not need_expand:
        return

    device = next(model_.parameters()).device

    # Expand token/user embeddings
    model_.poi_emb = _resize_embedding(model_.poi_emb, new_num_poi).to(device)
    model_.user_emb = _resize_embedding(model_.user_emb, new_num_user).to(device)

    # Expand positional embedding to at least current S
    if new_max_S > model_.pos_emb.num_embeddings:
        old_pos = model_.pos_emb
        model_.pos_emb = nn.Embedding(new_max_S, old_pos.embedding_dim).to(device)
        with torch.no_grad():
            # Copy old positions, init new tail
            model_.pos_emb.weight[: old_pos.num_embeddings].copy_(old_pos.weight)
            nn.init.normal_(
                model_.pos_emb.weight[old_pos.num_embeddings :], mean=0.0, std=0.02
            )

    # Expand output bias to match item vocabulary size (tied weights are poi_emb.weight)
    if new_num_poi > model_.out_bias.numel():
        old_bias = model_.out_bias
        new_bias = torch.zeros(new_num_poi, device=device, dtype=old_bias.dtype)
        with torch.no_grad():
            new_bias[: old_bias.numel()].copy_(old_bias)
        model_.out_bias = nn.Parameter(new_bias)

    log.info(
        "[STAN] resized: num_poi=%d, num_user=%d, max_seq_len=%d",
        model_.poi_emb.num_embeddings,
        model_.user_emb.num_embeddings,
        model_.pos_emb.num_embeddings,
    )


def _infer_vocab_sizes(dataloader: Any) -> Tuple[int, int, int]:
    """
    Scan train/val/test loaders to infer:
        - num_poi     : max POI_id (assume 1-based with 0 as padding) + 1
        - num_user    : max user_id (assume 1-based) + 1
        - max_seq_len : maximum sequence length across splits

    Returns:
        (num_poi_with_pad, num_user_with_pad, max_seq_len)
    """
    num_poi = 1
    num_user = 1
    max_seq_len = 1

    for loader in [
        dataloader.train_dataloader,
        dataloader.val_dataloader,
        dataloader.test_dataloader,
    ]:
        for batch in loader:
            poi = batch["POI_id"]  # [B, S]
            uid = batch["user_id"]  # [B]
            num_poi = max(num_poi, int(poi.max().item()))
            num_user = max(num_user, int(uid.max().item()))
            max_seq_len = max(max_seq_len, int(poi.size(1)))

    # +1 to include padding row at index 0
    return num_poi + 1, num_user + 1, max_seq_len


def _make_key_padding_mask(lengths: torch.Tensor, S: int) -> torch.Tensor:
    """
    Build key padding mask for Transformer.

    Args:
        lengths: [B] effective lengths per sample
        S      : int, sequence length

    Returns:
        mask: [B, S] bool, True indicates PAD positions to be masked.
    """
    device = lengths.device
    ar = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    valid = ar < lengths.unsqueeze(1)  # [B, S] True for valid tokens
    return ~valid  # [B, S] True for pads


def _time_delta_bins(
    timestamps: torch.Tensor, n_bins: int = 128, log_base: float = 1.6
) -> torch.Tensor:
    """
    Log-bucketize inter-event time gaps.

    Args:
        timestamps: [B, S] int64, unit: ns
        n_bins    : number of discrete bins
        log_base  : logarithm base for bucketing

    Returns:
        time_bins: [B, S] int64, first position set to 0, others are bucket IDs in [0, n_bins-1]
    """
    # Compute time gaps in seconds, clamp negatives to 0
    td = timestamps[:, 1:] - timestamps[:, :-1]  # [B, S-1] ns
    td = torch.clamp(td, min=0).to(torch.float32) / 1e9  # [B, S-1] s

    # Bucketization: bin = floor(log(1 + gap) / log_base)
    bins = torch.floor(torch.log1p(td) / math.log(log_base)).long()  # [B, S-1]
    bins = torch.clamp(bins, min=0, max=n_bins - 1)  # [B, S-1]

    # Prepend 0 for the first step (no previous event)
    zero_col = torch.zeros(
        (timestamps.size(0), 1), dtype=torch.long, device=timestamps.device
    )  # [B, 1]
    time_bins = torch.cat([zero_col, bins], dim=1)  # [B, S]
    return time_bins


# ============================== Model Core =================================
class STANEncoder(nn.Module):
    """
    Transformer-based encoder for next-POI prediction.

    Token representation:
        token = POI embedding + positional embedding + time-gap embedding + user bias

    Forward input shapes:
        poi_ids    : [B, S]
        user_ids   : [B]
        timestamps : [B, S]
        lengths    : [B]
    """

    def __init__(
        self,
        num_poi: int,
        num_user: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        time_bins: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.poi_emb = nn.Embedding(num_poi, d_model, padding_idx=0)  # [V, D]
        self.user_emb = nn.Embedding(num_user, d_model, padding_idx=0)  # [U, D]
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # [S_max, D]
        self.tgap_emb = nn.Embedding(time_bins, d_model)  # [Tbin, D]

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.drop = nn.Dropout(dropout)

        # Output bias for tied-weight output layer (logits = h @ E^T + b)
        self.out_bias = nn.Parameter(torch.zeros(num_poi))  # [V]

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.poi_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.tgap_emb.weight, mean=0.0, std=0.02)
        if self.poi_emb.padding_idx is not None:
            with torch.no_grad():
                self.poi_emb.weight[self.poi_emb.padding_idx].zero_()
        if self.user_emb.padding_idx is not None:
            with torch.no_grad():
                self.user_emb.weight[self.user_emb.padding_idx].zero_()

    def forward(
        self,
        poi_ids: torch.Tensor,  # [B, S]
        user_ids: torch.Tensor,  # [B]
        timestamps: torch.Tensor,  # [B, S]
        lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, V] unnormalized scores for all POIs
        """
        B, S = poi_ids.size()

        # Token embedding: POI
        tok = self.poi_emb(poi_ids)  # [B, S, D]

        # Positional embedding (indices 0..S-1 per batch)
        pos_idx = (
            torch.arange(S, device=poi_ids.device).unsqueeze(0).expand(B, S)
        )  # [B, S]
        pos = self.pos_emb(pos_idx)  # [B, S, D]
        tok = tok + pos  # [B, S, D]

        # Time-gap embedding (bucketized)
        t_bins = _time_delta_bins(
            timestamps, n_bins=self.tgap_emb.num_embeddings
        )  # [B, S]
        tok = tok + self.tgap_emb(t_bins)  # [B, S, D]

        # User bias added to all time steps
        u = self.user_emb(user_ids)[:, None, :]  # [B, 1, D]
        tok = tok + u  # [B, S, D]

        tok = self.drop(tok)  # [B, S, D]

        # Key padding mask for Transformer: True marks padding positions
        key_padding_mask = _make_key_padding_mask(lengths, S)  # [B, S]

        # Encode
        h = self.encoder(tok, src_key_padding_mask=key_padding_mask)  # [B, S, D]

        # Gather the last valid hidden state per sequence
        last_idx = torch.clamp(lengths - 1, min=0)  # [B]
        gather_idx = last_idx.view(B, 1, 1).expand(B, 1, h.size(-1))  # [B, 1, D]
        q = h.gather(dim=1, index=gather_idx).squeeze(1)  # [B, D]

        # Tied-weight output projection: logits = q @ E^T + b
        logits = q @ self.poi_emb.weight.t() + self.out_bias  # [B, V]
        return logits


# ============================== Train / Infer ===============================
def _build_model(dataloader: Any, **cfg: Any) -> nn.Module:
    """
    Construct model with vocab sizes inferred from dataloader.
    """
    global model, _device

    # Hyper-parameters from YAML (with defaults)
    d_model: int = int(model_args.get("d_model", 128))
    n_heads: int = int(model_args.get("n_heads", 4))
    n_layers: int = int(model_args.get("n_layers", 2))
    dropout: float = float(model_args.get("dropout", 0.1))
    time_bins: int = int(model_args.get("time_bins", 128))

    num_poi, num_user, max_seq_len = _infer_vocab_sizes(dataloader)
    log.info(
        "[STAN] vocab: num_poi=%d, num_user=%d, max_seq_len=%d",
        num_poi,
        num_user,
        max_seq_len,
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    built = STANEncoder(
        num_poi=num_poi,
        num_user=num_user,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        max_seq_len=max_seq_len,
        time_bins=time_bins,
    ).to(_device)

    model = built
    return built


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensors in (possibly nested) dict to device.
    """
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _get_targets(batch: Dict[str, Any]) -> torch.Tensor:
    """
    Extract ground-truth next-POI ID from batch.

    Returns:
        y: [B] long
    """
    y = batch["y_POI_id"]["POI_id"]  # [B]
    return y.long()


def init(dataloader: Any, **cfg: Any) -> None:
    """
    Called only when the outer pipeline enables 'val_while_train'.
    """
    _build_model(dataloader, **cfg)


def train_one_epoch(dataloader: Any, **cfg: Any) -> None:
    """
    Called only when the outer pipeline enables 'val_while_train'.
    """
    assert model is not None, "Call init(dataloader) before train_one_epoch()."
    _train_impl(dataloader, one_epoch=True, **cfg)


def train(dataloader: Any, **cfg: Any) -> None:
    """
    Standard training entry when 'val_while_train' is disabled.
    """
    if model is None:
        _build_model(dataloader, **cfg)
    _train_impl(dataloader, one_epoch=False, **cfg)


def _train_impl(dataloader: Any, one_epoch: bool = False, **cfg: Any) -> None:
    """
    Single/multi-epoch training loop.

    Notes:
        - Uses CE loss with ignore_index=0.
        - Supports on-the-fly capacity expansion for embeddings/pos-embedding/bias.
    """
    global model, _device
    assert model is not None and _device is not None

    model.train()

    # Training hyper-parameters
    epochs: int = int(model_args.get("epochs", 50))
    lr: float = float(model_args.get("lr", 1e-3))
    wd: float = float(model_args.get("weight_decay", 0.0))
    grad_clip: float = float(model_args.get("grad_clip", 1.0))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    train_loader = dataloader.train_dataloader

    run_epochs = 1 if one_epoch else epochs
    for ep in range(run_epochs):
        total_loss = 0.0
        total_cnt = 0

        for batch in train_loader:
            batch = _batch_to_device(batch, _device)

            # On-the-fly capacity expansion
            _maybe_expand_from_batch(batch, model)  # (no shape change for batch)

            # Fetch tensors
            poi_ids = batch["POI_id"].long()  # [B, S]
            user_ids = batch["user_id"].long()  # [B]
            timestamps = batch["timestamps"].long()  # [B, S]
            lengths = batch["mask"].long()  # [B]
            targets = _get_targets(batch)  # [B]

            # Forward + loss
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(poi_ids, user_ids, timestamps, lengths)  # [B, V]
                loss = F.cross_entropy(logits, targets, ignore_index=0)

            # Backward
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            # Track
            bs = poi_ids.size(0)
            total_loss += float(loss.item()) * bs
            total_cnt += bs

        avg = total_loss / max(1, total_cnt)
        log.info("[STAN][train] epoch %d/%d  loss=%.6f", ep + 1, run_epochs, avg)


@torch.no_grad()
def inference(dataloader: Any, **cfg: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inference on test set.

    Returns:
        preds: [N, V] probability distribution over POIs per sample
        gts  : [N]    ground-truth next-POI IDs
    """
    global model, _device
    if model is None:
        _build_model(dataloader, **cfg)
    assert model is not None

    device = _device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    test_loader = dataloader.test_dataloader

    for batch in test_loader:
        batch = _batch_to_device(batch, device)

        # On-the-fly capacity expansion (safe during eval as well)
        _maybe_expand_from_batch(batch, model)

        poi_ids = batch["POI_id"].long()  # [B, S]
        user_ids = batch["user_id"].long()  # [B]
        timestamps = batch["timestamps"].long()  # [B, S]
        lengths = batch["mask"].long()  # [B]
        targets = _get_targets(batch)  # [B]

        logits = model(poi_ids, user_ids, timestamps, lengths)  # [B, V]
        probs = torch.softmax(logits, dim=-1)  # [B, V]

        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())

    preds = torch.cat(all_probs, dim=0)  # [N, V]
    gts = torch.cat(all_targets, dim=0)  # [N]
    return preds, gts
