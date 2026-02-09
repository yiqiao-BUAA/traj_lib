import random
from typing import Optional, Tuple, Callable, Any
from collections.abc import Iterable
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from utils.dataloader.NPP.dataloader_base import BaseDataLoader
from utils.logger import get_logger
from utils.exargs import ConfigResolver
from utils.GPU_find import find_gpu

model_args = ConfigResolver("./model/FPMC/FPMC.yaml").parse()
logger = get_logger(__name__)
device = find_gpu()

pre_views = ["FPMC_count"]
post_views: list[str] = []
early_stop_func = 'half_improve'


def _to_long(x: torch.Tensor) -> torch.Tensor:
    return x.long() if x.dtype != torch.long else x


# ===================== FPMC model =====================
class FPMC(nn.Module):
    def __init__(self, n_user: int, n_item: int, n_factor: int = 64, pad_idx: int = 0):
        """
        n_user: user num (not include padding)
        n_item: item num (not include padding)
        n_factor: embedding dim
        pad_idx: padding idx in sequences (usually 0)
        """
        super().__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
        self.pad_idx = pad_idx

        # +1 for supporting 1-based ID and 0 as padding
        self.VUI = nn.Embedding(n_user + 1, n_factor, padding_idx=pad_idx)  # user-item
        self.VIU = nn.Embedding(n_item + 1, n_factor, padding_idx=pad_idx)  # item-user
        self.VIL = nn.Embedding(
            n_item + 1, n_factor, padding_idx=pad_idx
        )  # item-label (i in I, l in L)
        self.VLI = nn.Embedding(
            n_item + 1, n_factor, padding_idx=pad_idx
        )  # label-item (l in L, i in I)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for emb in (self.VUI, self.VIU, self.VIL, self.VLI):
            nn.init.xavier_uniform_(emb.weight)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].zero_()

    @staticmethod
    def _seq_context_mean(vli_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        vli_emb: (B, T, K) = VLI(seq)
        mask:    (B, T)    1/0
        return:  (B, K)    mean pooling
        """
        mask = mask.float()
        masked = vli_emb * mask.unsqueeze(-1)  # (B,T,K)
        lengths = mask.sum(dim=1).clamp_min(1.0)  # (B,)
        context = masked.sum(dim=1) / lengths.unsqueeze(-1)  # (B,K)
        return context

    def _make_mask(
        self, seq_padded: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Use seq to auto-generate mask; if external mask has same time dim, AND them.
        Return (B, T) 0/1 tensor
        """
        auto = seq_padded != self.pad_idx
        if mask is not None and mask.dim() == 2 and mask.size(1) == seq_padded.size(1):
            auto = auto & (mask > 0)
        return auto

    def forward(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        seq_padded: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score a batch of user-item pairs given their history.
        return:  (B,)
        users:      (B,)
        items:      (B,)
        seq_padded: (B,T)
        mask:       (B,T) 0/1 or None
        """
        # ---- to long ----
        users = _to_long(users)
        items = _to_long(items)
        seq_padded = _to_long(seq_padded)

        # --- make mask ---
        m = self._make_mask(seq_padded, mask)  # (B,T)

        # --- embeddings ---
        u = self.VUI(users)  # (B,K)
        i_ui = self.VIU(items)  # (B,K)
        i_il = self.VIL(items)  # (B,K)
        l_emb = self.VLI(seq_padded)  # (B,T,K)

        # --- mean pooling ---
        context = self._seq_context_mean(l_emb, m)  # (B,K)

        # --- FPMC scoring ---
        score = (u * i_ui).sum(dim=-1) + (i_il * context).sum(dim=-1)  # (B,)
        return score

    @torch.no_grad()
    def full_scores(
        self,
        users: torch.Tensor,
        seq_padded: torch.Tensor,
        mask: torch.Tensor | None = None,
        drop_padding_col: bool = True,
    ) -> torch.Tensor:
        """
        Input user batch and their history, score all items.
        Return (B, n_item); if drop_padding_col=False return (B, n_item+1)
        """
        # ---- to long ----
        users = _to_long(users)
        seq_padded = _to_long(seq_padded)

        # --- make mask ---
        m = self._make_mask(seq_padded, mask)  # (B,T)

        # --- user embedding ---
        U = self.VUI(users)  # (B,K)
        IU = self.VIU.weight  # (N,K)
        IL = self.VIL.weight  # (N,K)

        L_emb = self.VLI(seq_padded)  # (B,T,K)
        context = self._seq_context_mean(L_emb, m)  # (B,K)

        # term1: U @ IU^T -> (B,N)
        term1 = U @ IU.t()
        # term2: (IL @ context^T)^T -> (B,N)
        term2 = (IL @ context.t()).t()

        scores = term1 + term2  # (B,N)

        if drop_padding_col:
            scores = scores[:, 1:]
            # remove padding col, so item i∈[1..n_item] maps to col i-1 ∈ [0..n_item-1]
        return scores


model: Optional[FPMC] = None


# ===================== negative sampling =====================
def _sample_negative(
    pos_items: torch.Tensor, seq_padded: torch.Tensor, n_item: int, pad_idx: int = 0
) -> torch.Tensor:
    """
    Sample negative items for a batch of positive items and their history.
    Assume valid item ids ∈ [1, n_item], with 0 as padding.
    """
    pos_items = _to_long(pos_items)
    seq_padded = _to_long(seq_padded)

    device = pos_items.device
    B = pos_items.size(0)
    neg = torch.empty(B, dtype=torch.long, device=device)

    seq_np = seq_padded.detach().cpu().tolist()
    pos_np = pos_items.detach().cpu().tolist()
    forbid = []
    for b in range(B):
        s = set(x for x in seq_np[b] if x != pad_idx)
        s.add(pos_np[b])
        s.add(pad_idx)
        forbid.append(s)

    for b in range(B):
        while True:
            j = random.randint(1, n_item)
            if j not in forbid[b]:
                neg[b] = j
                break
    return neg


# ===================== Train =====================
def train(
    train_dl: DataLoader,
    val_dl: DataLoader,
    view_value: dict[str, Any],
    eval_funcs: dict[str, Callable],
    **kwargs,
) -> Iterable[dict[str, Any]]:
    global model

    n_user, n_item = view_value["n_user"], view_value["n_item"]
    logger.info(f"n_user: {n_user}, n_item: {n_item}")

    model = FPMC(
        n_user, n_item, n_factor=model_args["n_factor"], pad_idx=model_args["pad_idx"]
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=model_args["lr"], weight_decay=model_args["weight_decay"]
    )

    model.train()
    for epoch in range(model_args["epochs"]):
        epoch_loss = 0.0
        total = 0
        for batch in tqdm(train_dl):
            users = _to_long(batch["user_id"].to(device))
            seq_padded = _to_long(batch["POI_id"].to(device))
            mask = batch["mask"].to(device) if "mask" in batch else None
            pos_items = _to_long(batch["y_POI_id"]["POI_id"].to(device))

            neg_items = _sample_negative(pos_items, seq_padded, n_item, pad_idx=0)

            s_pos = model(users, pos_items, seq_padded, mask)  # (B,)
            s_neg = model(users, neg_items, seq_padded, mask)  # (B,)

            loss = F.softplus(-(s_pos - s_neg)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = users.size(0)
            epoch_loss += loss.item() * bs
            total += bs

        avg = epoch_loss / max(total, 1)
        logger.info(f"Epoch {epoch+1:02d} - BPR-loss = {avg:.4f}")

        inference_res = inference(val_dl, view_value)
        pred = inference_res['pred']
        gt = inference_res['gts']
        scores = {}
        for name, func in eval_funcs.items():
            score = func(pred, gt)
            scores[name] = score

        yield [scores, {'loss': avg, 'title': 'train loss'}]

    logger.info("Training complete ✔️")


# ===================== Inference =====================
@torch.no_grad()
def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return all-item scores for each sample in val/test set.
    Return:
      - preds: numpy.ndarray, (N_sample, n_item)
      - gts:   numpy.ndarray, (N_sample,)
    """
    assert model is not None
    model.eval()

    batch_scores_list = []
    batch_gt_idx_list = []

    for batch in tqdm(test_dl):
        users = _to_long(batch["user_id"].to(device))
        seq_padded = _to_long(batch["POI_id"].to(device))
        mask = batch["mask"].to(device) if "mask" in batch else None
        gts_id = _to_long(batch["y_POI_id"]["POI_id"].to(device))

        # (B, n_item)  remove padding col so item i∈[1..n_item] maps to col i-1 ∈ [0..n_item-1]
        scores = model.full_scores(users, seq_padded, mask)  # (B, n_item)

        # change ground-truth item IDs to column indices: i∈[1..n_item] -> i-1 ∈ [0..n_item-1]
        gt_idx = (gts_id - 1).clamp(min=0, max=model.n_item - 1)

        batch_scores_list.append(scores.detach().cpu().numpy())  # (B, n_item)
        batch_gt_idx_list.append(gt_idx.detach().cpu().numpy())  # (B,)

    preds = np.concatenate(batch_scores_list, axis=0)  # (N, n_item)
    gts = np.concatenate(batch_gt_idx_list, axis=0)  # (N,)

    return {'pred': preds, 'gts': gts}
