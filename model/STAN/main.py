# model/STAN/main.py
# -*- coding: utf-8 -*-
"""
PyTorch implementation of the STAN model, refactored to align with the paper:
"STAN: Spatio-Temporal Attention Network for Next Location Recommendation".

Key changes from the original user implementation:
1.  **Bi-Attention Architecture**: Implemented the two-layer attention system:
    - SelfAttentionAggregationLayer: Fuses spatio-temporal distance matrices directly
      into the attention score calculation.
    - AttentionMatchingLayer: Matches all candidates against all historical items to
      preserve Personalized Item Frequency (PIF).
2.  **Explicit Spatio-Temporal Matrices**: Added on-the-fly calculation of (N, S, S)
    spatio-temporal distance matrices for every batch, as described in the paper.
3.  **Balanced Sampler Loss**: The training loop now uses a balanced sampler to compute
    the loss, based on one positive and `s` negative samples per instance.
4.  **Data Adaptation Layer**: The training/inference loops now adapt the dataloader's
    output format (separate 'latitude', 'longitude') to the model's expected
    input format ('gps_locations' tensor) internally, respecting read-only
    dataloader files.
"""

import math
from typing import Any, Dict, Tuple, Optional, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
pre_views: list[str] = []
post_views: list[str] = []

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
    """
    if new_num <= old_emb.num_embeddings:
        return old_emb

    new_emb = nn.Embedding(
        new_num, old_emb.embedding_dim, padding_idx=old_emb.padding_idx
    )
    with torch.no_grad():
        nn.init.normal_(new_emb.weight, mean=0.0, std=init_std)
        new_emb.weight[: old_emb.num_embeddings].copy_(old_emb.weight)
        if old_emb.padding_idx is not None:
            new_emb.weight[old_emb.padding_idx].zero_()
    return new_emb


def _maybe_expand_from_batch(batch: Dict[str, Any], model_: nn.Module) -> None:
    """
    Expand embeddings on the fly based on the current batch.
    """
    poi_max: int = int(batch["POI_id"].max().item())
    user_max: int = int(batch["user_id"].max().item())

    need_expand = False
    new_num_poi = model_.poi_emb.num_embeddings
    new_num_user = model_.user_emb.num_embeddings

    if poi_max + 1 > new_num_poi:
        new_num_poi = poi_max + 1
        need_expand = True
    if user_max + 1 > new_num_user:
        new_num_user = user_max + 1
        need_expand = True

    if not need_expand:
        return

    device = next(model_.parameters()).device
    model_.poi_emb = _resize_embedding(model_.poi_emb, new_num_poi).to(device)
    model_.user_emb = _resize_embedding(model_.user_emb, new_num_user).to(device)

    log.info(
        "[STAN] resized: num_poi=%d, num_user=%d",
        model_.poi_emb.num_embeddings,
        model_.user_emb.num_embeddings,
    )

def _calculate_time_delta_matrix(timestamps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the absolute time difference matrix between all pairs of check-ins.
    Args:
        timestamps: [B, S] tensor of timestamps for each check-in.
    Returns:
        delta_t: [B, S, S] tensor where delta_t[b, i, j] = |t_i - t_j|.
    """
    # [B, S, 1] - [B, 1, S] -> [B, S, S]
    delta_t = torch.abs(timestamps.unsqueeze(2) - timestamps.unsqueeze(1))
    delta_t_hours = delta_t.float() / 3600.0  # Convert seconds to hours
    return torch.clamp(delta_t_hours, max=500.0)


def _calculate_haversine_distance_matrix(gps_locations: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Haversine distance matrix between all pairs of GPS coordinates.
    Args:
        gps_locations: [B, S, 2] tensor with (latitude, longitude) for each check-in.
    Returns:
        delta_s: [B, S, S] tensor where delta_s[b, i, j] is the distance in km.
    """
    R = 6371  # Earth radius in kilometers
    # [B, S, 2] -> lat/lon [B, S]
    lat = torch.deg2rad(gps_locations[..., 0])
    lon = torch.deg2rad(gps_locations[..., 1])

    # [B, S, 1] - [B, 1, S] -> [B, S, S]
    dlat = lat.unsqueeze(2) - lat.unsqueeze(1)
    dlon = lon.unsqueeze(2) - lon.unsqueeze(1)

    # Haversine formula
    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat.unsqueeze(2))
        * torch.cos(lat.unsqueeze(1))
        * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
    distance = R * c
    distance_in_hm = distance / 100.0 # Convert km to hundred meters as basic unit
    return torch.clamp(distance_in_hm, max=1000.0)


def _make_key_padding_mask(lengths: torch.Tensor, S: int) -> torch.Tensor:
    """
    Build key padding mask for Transformer. True indicates PAD positions.
    """
    device = lengths.device
    ar = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    return ar >= lengths.unsqueeze(1)  # [B, S]


# ============================== Model Core =================================
class SpatioTemporalEmbedding(nn.Module):
    """
    Embeds continuous spatio-temporal distances into dense vectors.
    Implements the simpler variant from the paper (Eq. 3).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.e_delta_t = nn.Parameter(torch.randn(d_model))
        self.e_delta_s = nn.Parameter(torch.randn(d_model))

    def forward(self, delta_t: torch.Tensor, delta_s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_t: [B, S, S] time difference matrix.
            delta_s: [B, S, S] spatial distance matrix.
        Returns:
            Combined spatio-temporal embedding [B, S, S, D].
        """
        # [B, S, S, 1] * [D] -> [B, S, S, D]
        t_emb = delta_t.unsqueeze(-1) * self.e_delta_t
        s_emb = delta_s.unsqueeze(-1) * self.e_delta_s
        return t_emb + s_emb


class SelfAttentionAggregationLayer(nn.Module):
    """
    First layer of STAN: Aggregates historical check-ins using self-attention
    modulated by spatio-temporal distances.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Spatio-temporal embedding projection
        self.w_st = nn.Linear(d_model, n_heads, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, st_emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input sequence embeddings [B, S, D].
            st_emb: Spatio-temporal embeddings [B, S, S, D].
            mask: Key padding mask [B, S].
        Returns:
            Updated sequence representations [B, S, D].
        """
        B, S, D = x.shape
        residual = x

        q = self.w_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v are now [B, H, S, D_h]

        # Project spatio-temporal embeddings to match attention heads
        # [B, S, S, D] -> [B, S, S, H] -> [B, H, S, S]
        st_bias = self.w_st(st_emb).permute(0, 3, 1, 2)

        # Calculate attention scores with spatio-temporal bias
        # [B, H, S, D_h] @ [B, H, D_h, S] -> [B, H, S, S]
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores + st_bias  # Add spatio-temporal effect

        if mask is not None:
            # Mask needs to be broadcastable: [B, S] -> [B, 1, 1, S]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1) # [B, H, S, S]
        attn_weights = self.dropout(attn_weights)

        # [B, H, S, S] @ [B, H, S, D_h] -> [B, H, S, D_h]
        attn_output = attn_weights @ v
        # -> [B, S, H, D_h] -> [B, S, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        attn_output = self.w_o(attn_output)

        return self.layer_norm(residual + self.dropout(attn_output))


class AttentionMatchingLayer(nn.Module):
    """
    Second layer of STAN: Matches all candidate POIs against the updated
    historical representations to compute final scores.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, candidates_emb: torch.Tensor, history_reps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            candidates_emb: All POI embeddings [L, D].
            history_reps: Updated history representations from aggregator [B, S, D].
            mask: Key padding mask [B, S].
        Returns:
            Logits for all candidates [B, L].
        """
        # [B, S, D] -> [B, D, S]
        history_reps_t = history_reps.transpose(1, 2)

        # [L, D] @ [B, D, S] -> [B, L, S] (using broadcasting for batch)
        # This computes the matching score of each candidate with each history item.
        matching_scores = candidates_emb @ history_reps_t

        if mask is not None:
            # Mask out contributions from padded history items
            # [B, S] -> [B, 1, S]
            masked_scores = matching_scores.masked_fill(mask.unsqueeze(1), 0.0)

            # [B, L, S] -> [B, L]
            summed_scores = masked_scores.sum(dim=2)

            # [B] -> [B, 1]
            valid_lengths = mask.shape[1] - mask.sum(dim=1).clamp(min=1.0)
            
            # [B, L] / [B, 1] -> [B, L]
            final_logits = summed_scores / valid_lengths.unsqueeze(1)
        else:
            final_logits = matching_scores.mean(dim=2)
        
        return final_logits


class STAN(nn.Module):
    """
    Main STAN model, orchestrating the bi-attention architecture.
    """
    def __init__(
        self,
        num_poi: int,
        num_user: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.poi_emb = nn.Embedding(num_poi, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_user, d_model, padding_idx=0)
        
        self.st_embedding = SpatioTemporalEmbedding(d_model)
        
        self.aggregator_layers = nn.ModuleList([
            SelfAttentionAggregationLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.matching_layer = AttentionMatchingLayer()
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.poi_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)
        if self.poi_emb.padding_idx is not None:
            with torch.no_grad():
                self.poi_emb.weight[self.poi_emb.padding_idx].zero_()
        if self.user_emb.padding_idx is not None:
            with torch.no_grad():
                self.user_emb.weight[self.user_emb.padding_idx].zero_()

    def forward(
        self,
        poi_ids: torch.Tensor,      # [B, S]
        user_ids: torch.Tensor,     # [B]
        timestamps: torch.Tensor,   # [B, S]
        gps_locations: torch.Tensor,# [B, S, 2] <- Model's internal expectation
        lengths: torch.Tensor,      # [B]
    ) -> torch.Tensor:
        """
        Returns:
            logits: [B, L] unnormalized scores for all POIs. L is total num_poi.
        """
        B, S = poi_ids.size()
        
        # 1. Create masks
        key_padding_mask = _make_key_padding_mask(lengths, S) # [B, S]

        # 2. Calculate Spatio-Temporal matrices and embed them
        delta_t = _calculate_time_delta_matrix(timestamps) # [B, S, S]
        delta_s = _calculate_haversine_distance_matrix(gps_locations) # [B, S, S]
        st_emb = self.st_embedding(delta_t, delta_s) # [B, S, S, D]

        # 3. Create initial embeddings
        # [B, S, D] + [B, 1, D] -> [B, S, D]
        x = self.poi_emb(poi_ids) + self.user_emb(user_ids).unsqueeze(1)
        x = self.dropout(x)

        # 4. Self-Attention Aggregation Layer
        for layer in self.aggregator_layers:
            x = layer(x, st_emb, key_padding_mask) # [B, S, D]

        # 5. Attention Matching Layer
        # Use all POI embeddings as candidates
        all_poi_embeddings = self.poi_emb.weight # [L, D]
        logits = self.matching_layer(all_poi_embeddings, x, key_padding_mask) # [B, L]
        
        return logits


# ============================== Train / Infer ===============================
def _build_model(
    train_dl: DataLoader,
    val_dl: Optional[DataLoader],
    test_dl: Optional[DataLoader],
    **cfg: Any,
) -> nn.Module:
    global model, model_args, _device

    # Infer vocab sizes (assuming padding=0)
    num_poi, num_user = 1, 1
    for loader in (train_dl, val_dl, test_dl):
        if loader is None: continue
        for batch in loader:
            num_poi = max(num_poi, int(batch["POI_id"].max().item()))
            num_user = max(num_user, int(batch["user_id"].max().item()))
    num_poi += 1
    num_user += 1
    
    log.info("[STAN] vocab: num_poi=%d, num_user=%d", num_poi, num_user)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    built = STAN(
        num_poi=num_poi,
        num_user=num_user,
        d_model=model_args["d_model"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        dropout=model_args["dropout"],
    ).to(_device)

    model = built
    return built


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
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
    return batch["y_POI_id"]["POI_id"].long()


def init(dataloader: Any, **cfg: Any) -> None:
    _build_model(dataloader, **cfg)


def train_one_epoch(dataloader: Any, **cfg: Any) -> None:
    assert model is not None, "Call init(dataloader) before train_one_epoch()."
    _train_impl(dataloader, one_epoch=True, **cfg)


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    view_value: Dict[str, Any],
    eval_funcs: Optional[
        Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
    ] = None,
    **cfg: Any,
) -> Iterable[Dict[str, float]]:
    global model
    if model is None:
        _build_model(train_dataloader, val_dataloader, None, **cfg)

    return _train_impl(
        train_dataloader,
        val_dataloader,
        view_value,
        one_epoch=False,
        eval_funcs=eval_funcs,
        **cfg,
    )


def _train_impl(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    view_value: Dict[str, Any],
    one_epoch: bool = False,
    eval_funcs: Optional[
        Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
    ] = None,
    **cfg: Any,
) -> Iterable[Dict[str, float]]:
    global model, _device
    assert model is not None and _device is not None
    model.train()

    grad_clip: float = float(model_args["grad_clip"])
    s: int = int(model_args["num_negative_samples"])

    opt = torch.optim.AdamW(
        model.parameters(), lr=model_args["lr"], weight_decay=model_args["weight_decay"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    run_epochs: int = 1 if one_epoch else model_args["epochs"]

    for ep in range(run_epochs):
        total_loss, total_cnt = 0.0, 0
        for batch in train_dataloader:
            batch = _batch_to_device(batch, _device)
            _maybe_expand_from_batch(batch, model)

            poi_ids = batch["POI_id"].long()
            user_ids = batch["user_id"].long()
            timestamps = batch["timestamps"].long()
            lengths = batch["mask"].long()
            targets = _get_targets(batch)

            # =================== START: DATA ADAPTATION ===================
            # 此处是核心修改。我们检查批次数据中是否存在 'latitude' 和 'longitude'
            # 如果存在，则将它们合并为模型所需的 'gps_locations' 张量。
            if 'latitude' in batch and 'longitude' in batch:
                lat = batch['latitude'].unsqueeze(2)    # 形状: [B, S] -> [B, S, 1]
                lon = batch['longitude'].unsqueeze(2)   # 形状: [B, S] -> [B, S, 1]
                gps_locations = torch.cat((lat, lon), dim=2).float() # 形状: [B, S, 2]
            else:
                # 如果批次数据中连独立的经纬度信息都没有，则无法继续，必须报错。
                raise KeyError("Batch is missing required 'latitude' and/or 'longitude' tensors.")
            # ==================== END: DATA ADAPTATION ====================

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # 将新创建的 gps_locations 张量传递给模型
                logits = model(poi_ids, user_ids, timestamps, gps_locations, lengths)
                
                # Balanced Sampler Loss Implementation
                B, L = logits.shape
                
                # [B] -> [B, 1]
                pos_indices = targets.unsqueeze(1)
                # [B, L] -> [B, 1]
                pos_logits = logits.gather(1, pos_indices)

                # Sample `s` negative items per instance in the batch
                neg_indices = torch.randint(1, L, (B, s), device=_device)
                # Ensure no collision with positive samples (rare, but good practice)
                collision = (neg_indices == pos_indices)
                while torch.any(collision):
                    neg_indices[collision] = torch.randint(
                        1, L, (collision.sum(),), device=_device
                    )
                    collision = (neg_indices == pos_indices)
                
                # [B, L] -> [B, S]
                neg_logits = logits.gather(1, neg_indices)

                # Binary cross-entropy style loss
                pos_loss = F.logsigmoid(pos_logits)
                neg_loss = F.logsigmoid(-neg_logits).sum(dim=1, keepdim=True)
                
                loss = -(pos_loss + neg_loss).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            bs = poi_ids.size(0)
            total_loss += loss.item() * bs
            total_cnt += bs

        avg_loss = total_loss / max(1, total_cnt)
        log.info("[STAN][train] epoch %d/%d  loss=%.6f", ep + 1, run_epochs, avg_loss)

        if eval_funcs:
            inference_res = _inference_on_split(val_dataloader, view_value)
            preds = inference_res['pred']
            gts = inference_res['gts']
            scores: Dict[str, float] = {
                name: float(fn(preds, gts)) for name, fn in eval_funcs.items()
            }

            score_str = "  ".join([f"{name}={value:.4f}" for name, value in scores.items()])
            log.info("[STAN][val]   epoch %d/%d  %s", ep + 1, run_epochs, score_str)

            yield [{**scores}, {"loss": avg_loss}]
        else:
            log.info("[STAN][val]   epoch %d/%d  %s", ep + 1, run_epochs)
            yield {"loss": avg_loss}


@torch.no_grad()
def _inference_on_split(
    split_dataloader: DataLoader,
    view_value: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    global model, _device
    assert model is not None
    device = _device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    probs_list, targets_list = [], []
    for batch in split_dataloader:
        batch = _batch_to_device(batch, device)
        _maybe_expand_from_batch(batch, model)

        poi_ids = batch["POI_id"].long()
        user_ids = batch["user_id"].long()
        timestamps = batch["timestamps"].long()
        lengths = batch["mask"].long()
        targets = _get_targets(batch)
        
        # =================== START: DATA ADAPTATION ===================
        # 在推理/验证阶段也进行同样的数据适配
        if 'latitude' in batch and 'longitude' in batch:
            lat = batch['latitude'].unsqueeze(2)
            lon = batch['longitude'].unsqueeze(2)
            gps_locations = torch.cat((lat, lon), dim=2).float()
        else:
            raise KeyError("Batch is missing required 'latitude' and/or 'longitude' tensors for inference.")
        # ==================== END: DATA ADAPTATION ====================

        logits = model(poi_ids, user_ids, timestamps, gps_locations, lengths)
        probs = torch.softmax(logits, dim=-1)

        probs_list.append(probs.cpu())
        targets_list.append(targets.cpu())

    return {'pred': torch.cat(probs_list, dim=0), 'gts': torch.cat(targets_list, dim=0)}


@torch.no_grad()
def inference(
    test_dataloader: DataLoader, view_value: Dict[str, Any], **cfg: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    global model
    if model is None:
        # NOTE: For inference, a train_dl might be needed by _build_model
        # to see the full vocabulary. Passing test_dataloader as a stand-in.
        _build_model(test_dataloader, None, test_dataloader, **cfg)
    return _inference_on_split(test_dataloader, view_value)