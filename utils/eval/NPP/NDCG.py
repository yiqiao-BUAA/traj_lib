from tqdm import tqdm
import numpy as np
import torch

from utils.register import register_eval
from utils.logger import get_logger

logger = get_logger(__name__)

from typing import Union
import torch
import numpy as np

def evaluate(
    preds: Union[torch.Tensor, np.ndarray],
    gts:   Union[torch.Tensor, np.ndarray],
    topk: int = 20
) -> float:
    """
    Computes NDCG@K for single-label targets.

    Args:
        preds: [B, N] scores/probabilities per item.
        gts:   [B]    ground-truth class index per row (0..N-1).
        topk:  K in NDCG@K.

    Returns:
        float: mean NDCG@K over the batch.

    Notes:
        For single positive label per row, IDCG = 1, so NDCG@K == DCG@K.
    """
    # --- checks ---
    if not isinstance(preds, (torch.Tensor, np.ndarray)) or not isinstance(gts, (torch.Tensor, np.ndarray)):
        raise TypeError("preds and gts must be torch.Tensor or np.ndarray.")
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D [B, N], got shape {getattr(preds, 'shape', None)}.")
    if gts.ndim != 1 or len(gts) != len(preds):
        raise ValueError("gts must be 1D [B] and have the same length as preds.")
    if topk <= 0:
        raise ValueError("topk must be a positive integer.")

    B, N = int(preds.shape[0]), int(preds.shape[1])
    K = int(min(topk, N))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elem_size = preds.element_size() if isinstance(preds, torch.Tensor) else preds.dtype.itemsize
    rows_per_chunk = max(1, 1_073_741_824 // max(1, N * elem_size))

    discount_full = 1.0 / torch.log2(torch.arange(2, K + 2, device=device, dtype=torch.float32))  # 1/log2(1+rank)

    ndcg_list = []
    with torch.no_grad():
        # for start in tqdm(range(0, B, rows_per_chunk), desc=f"Evaluating NDCG   {topk:>2}", unit="chunk"):
        for start in range(0, B, rows_per_chunk):  # disable tqdm for cleaner logging
            end = min(start + rows_per_chunk, B)

            p_slice = preds[start:end]
            g_slice = gts[start:end]

            p_t = p_slice if isinstance(p_slice, torch.Tensor) else torch.as_tensor(p_slice)
            g_t = g_slice if isinstance(g_slice, torch.Tensor) else torch.as_tensor(g_slice)

            p_t = p_t.to(device=device)
            g_t = g_t.to(device=device, dtype=torch.long)

            topk_idx = p_t.topk(k=K, dim=1).indices          # [chunk, K]

            hits = (topk_idx == g_t.unsqueeze(1))            # [chunk, K], bool
            dcg = (hits.float() * discount_full).sum(dim=1)  # [chunk]

            ndcg = dcg  # / 1.0
            ndcg_list.append(ndcg.cpu())

    return torch.cat(ndcg_list, dim=0).mean().item()


@register_eval("NDCG1")
def ndcg1(
    preds: np.ndarray,
    gts: np.ndarray,
    topk: int = 1
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("NDCG5")
def ndcg5(
    preds: np.ndarray,
    gts: np.ndarray,
    topk: int = 5
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("NDCG10")
def ndcg10(
    preds: np.ndarray,
    gts: np.ndarray,
    topk: int = 10
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("NDCG20")
def ndcg20(
    preds: np.ndarray,
    gts: np.ndarray,
    topk: int = 20
) -> float:
    return evaluate(preds, gts, topk=topk)
