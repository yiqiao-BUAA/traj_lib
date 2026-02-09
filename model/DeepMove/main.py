from typing import Any, Tuple, Callable
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils.logger import get_logger
from utils.exargs import ConfigResolver, ParseDict

model_args = ConfigResolver("./model/DeepMove/DeepMove.yaml").parse()

model = None
pre_views = ["DeepMove_preview"]
post_views = ["DeepMove_postview"]

logger = get_logger(__name__)

def _cpu_1d_int64(x: torch.Tensor | np.ndarray | list[int] | int | None) -> torch.Tensor | None:
    """Return CPU 1D int64 tensor for pack_padded_sequence lengths."""
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.detach().cpu()
        if t.dim() == 0:
            t = t.view(1)
        return t.to(torch.int64)
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if t.dim() == 0:
            t = t.view(1)
        return t.to(torch.int64)
    if isinstance(x, (list, tuple)):
        return torch.tensor([int(v) for v in x], dtype=torch.int64)
    if isinstance(x, int):
        return torch.tensor([x], dtype=torch.int64)
    raise TypeError(f"Unsupported lengths type: {type(x)}")

def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors to device, keeping length-like keys on CPU.
    Keys containing 'len' or 'length' (case-insensitive) are kept on CPU.
    """
    if torch.is_tensor(batch):
        return batch.to(device)

    if isinstance(batch, dict):
        out: dict[Any, Any] = {}
        for k, v in batch.items():
            kl = str(k).lower()
            if "len" in kl or "length" in kl:
                # 长度相关：保持在 CPU（不做 tolist 转换，和你原逻辑一致）
                out[k] = v.detach().cpu() if torch.is_tensor(v) else v
            else:
                out[k] = move_to_device(v, device)
        return out

    if isinstance(batch, tuple):
        return tuple(move_to_device(v, device) for v in batch)

    if isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]

    return batch



class Attn(nn.Module):
    """Attention Module (dot/general)."""

    def __init__(self, method: str, hidden_size: int) -> None:
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == "concat":
            # 未使用 concat，保留占位
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        # out_state: (B, S, H); history: (B, T, H)
        if self.method == "dot":
            attn_energies = torch.bmm(out_state, history.permute(0, 2, 1))
        elif self.method == "general":
            attn_energies = torch.bmm(out_state, self.attn(history).permute(0, 2, 1))
        else:
            raise ValueError(f"Unsupported attn method: {self.method}")
        return F.softmax(attn_energies, dim=2)


class DeepMove(nn.Module):
    """RNN model with long-term history attention (device-agnostic)."""

    def __init__(self, config: ParseDict, view_value: dict[str, Any]) -> None:
        super().__init__()
        self.loc_size = view_value["loc_size"]
        self.loc_emb_size = config["loc_emb_size"]
        self.tim_size = view_value["tim_size"]
        self.tim_emb_size = config["tim_emb_size"]
        self.hidden_size = config["hidden_size"]
        self.attn_type = config["attn_type"]
        self.rnn_type = config["rnn_type"]

        self.emb_loc = nn.Embedding(
            self.loc_size,
            self.loc_emb_size,
            padding_idx=view_value["loc_pad"],
        )
        self.emb_tim = nn.Embedding(
            self.tim_size,
            self.tim_emb_size,
            padding_idx=view_value["tim_pad"],
        )

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)

        self.rnn_encoder: nn.Module
        self.rnn_decoder: nn.Module
        if self.rnn_type == "GRU":
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1, batch_first=True)
        elif self.rnn_type == "LSTM":
            self.rnn_encoder = nn.LSTM(
                input_size, self.hidden_size, 1, batch_first=True
            )
            self.rnn_decoder = nn.LSTM(
                input_size, self.hidden_size, 1, batch_first=True
            )
        elif self.rnn_type == "RNN":
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1, batch_first=True)
            self.rnn_decoder = nn.LSTM(
                input_size, self.hidden_size, 1, batch_first=True
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")
        
        assert self.rnn_encoder is not None and self.rnn_decoder is not None

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=config["dropout_p"])
        self.init_weights()

    def init_weights(self) -> None:
        ih = (
            param.data for name, param in self.named_parameters() if "weight_ih" in name
        )
        hh = (
            param.data for name, param in self.named_parameters() if "weight_hh" in name
        )
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    @staticmethod
    def _ensure_batch2(x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has shape (B, S). If (S,), add batch dim."""
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    @staticmethod
    def _lengths_from_pad(seq: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Compute per-sample lengths (B,) on CPU using pad_id. Clamp to >=1 to avoid empty sequences."""
        B, S = seq.shape
        if pad_id is None:
            lens = torch.full((B,), S, dtype=torch.int64)
        else:
            lens = (seq != pad_id).sum(dim=1).to(torch.int64)
        if (lens == 0).any():
            n0 = int((lens == 0).sum())
            # logger.warning(
            #     f"[lengths] found {n0} zero-length sequences; clamping to 1 to proceed."
            # )
            lens = torch.clamp(lens, min=1)
        return lens.cpu()

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:

        # Inputs (expect indices)
        loc_seq = batch["POI_id"]
        tim_seq = batch["timestamps"]
        # Optional history fields
        history_loc = batch.get("history_POI_id", loc_seq)
        history_tim = batch.get("history_timestamps", tim_seq)

        # Embedding & concat (batch_first)
        x = torch.cat(
            (self.emb_loc(loc_seq), self.emb_tim(tim_seq)), dim=2
        )  # (B, S, I)
        x = self.dropout(x)
        history_x = torch.cat(
            (self.emb_loc(history_loc), self.emb_tim(history_tim)), dim=2
        )  # (B, T, I)
        history_x = self.dropout(history_x)

        # Hidden states on same device as weights
        B = x.size(0)
        h1 = self.emb_loc.weight.new_zeros(1, B, self.hidden_size)
        h2 = self.emb_loc.weight.new_zeros(1, B, self.hidden_size)
        c1 = self.emb_loc.weight.new_zeros(1, B, self.hidden_size)
        c2 = self.emb_loc.weight.new_zeros(1, B, self.hidden_size)

        # Lengths (CPU int64)
        assert self.emb_loc.padding_idx is not None
        loc_len = self._lengths_from_pad(loc_seq, self.emb_loc.padding_idx)
        hist_len = self._lengths_from_pad(history_loc, self.emb_loc.padding_idx)

        # Pack (batch_first=True)
        loc_len_1d = _cpu_1d_int64(loc_len)
        hist_len_1d = _cpu_1d_int64(hist_len)
        assert loc_len_1d is not None and hist_len_1d is not None
        pack_x = pack_padded_sequence(
            x, lengths=loc_len_1d, batch_first=True, enforce_sorted=False
        )
        pack_history_x = pack_padded_sequence(
            history_x,
            lengths=hist_len_1d,
            batch_first=True,
            enforce_sorted=False,
        )

        # Encode / Decode
        if self.rnn_type in ("GRU", "RNN"):
            hidden_history, h1 = self.rnn_encoder(pack_history_x, h1)
            hidden_state, h2 = self.rnn_decoder(pack_x, h2)
        elif self.rnn_type == "LSTM":
            hidden_history, (h1, c1) = self.rnn_encoder(pack_history_x, (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(pack_x, (h2, c2))

        # Unpack (batch_first=True) -> (B, T/S, H)
        hidden_history, _ = pad_packed_sequence(hidden_history, batch_first=True)
        hidden_state, _ = pad_packed_sequence(hidden_state, batch_first=True)

        # Attention & context
        attn_weights = self.attn(hidden_state, hidden_history)  # (B, S, T)
        context = attn_weights.bmm(hidden_history)  # (B, S, H)
        out = torch.cat((hidden_state, context), dim=2)  # (B, S, 2H)

        # Take last valid timestep per sample
        last_t = (loc_len - 1).to(device=out.device)
        out = out[torch.arange(B, device=out.device), last_t, :]  # (B, 2H)
        out = self.dropout(out)

        # Classifier
        y = self.fc_final(out)  # (B, loc_size)
        score = F.log_softmax(y, dim=1)
        return score

    def calculate_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        assert self.emb_loc.padding_idx is not None
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.emb_loc.padding_idx, reduction="mean"
        )
        scores = self.forward(batch)
        target = batch["y_POI_id"]["POI_id"].to(scores.device).long()
        return criterion(scores, target)


def train(
    train_dl: DataLoader,
    val_dl: DataLoader,
    view_value: dict[str, Any],
    eval_funcs: dict[str, Callable],
    **kwargs,
) -> Iterable[dict[str, Any]]:
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepMove(model_args, view_value).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_args["learning_rate"])

    for epoch in range(model_args["epochs"]):
        model.train()
        loss_list = []
        for batch in train_dl:
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            assert isinstance(batch, dict)
            loss = model.calculate_loss(batch)
            loss.backward()
            # _print_grad_stats(model)
            optimizer.step()
            loss_list.append(loss.item())
        logger.info(
            f"{epoch + 1}/{model_args['epochs']} - Loss: {sum(loss_list) / len(loss_list):.6f}"
        )
        inference_res = inference(val_dl, view_value)
        scores = {}
        for name, func in eval_funcs.items():
            score = func(inference_res['pred'], inference_res['gts'])
            scores[name] = score

        yield [scores, {'loss': sum(loss_list) / len(loss_list)}]

@torch.no_grad()
def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Recall for the predictions.
    preds : [batch, n_items] as the probability of each item
    gts : [batch]
    """
    assert model is not None
    device = next(model.parameters()).device
    model.eval()
    prob_chunks, gt_chunks = [], []
    with torch.no_grad():
        for batch in test_dl:
            batch = move_to_device(batch, device)
            assert isinstance(batch, dict)
            log_scores = model.forward(batch)
            probs = log_scores.exp().detach().cpu()
            prob_chunks.append(probs)
            gt_chunks.append(batch["y_POI_id"]["POI_id"].detach().cpu())
    preds = torch.cat(prob_chunks, dim=0).numpy()  # [batch, n_items]
    gts = torch.cat(gt_chunks, dim=0).numpy()  # [batch]
    return {'pred': preds, 'gts': gts}
