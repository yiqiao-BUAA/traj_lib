from typing import Union
import torch
import numpy as np

from utils.register import register_eval
from utils.logger import get_logger

logger = get_logger(__name__)

def evaluate_mrr(
    preds: Union[torch.Tensor, np.ndarray],
    gts:   Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Computes MRR (not truncated) for single-label targets.

    Args:
        preds: [B, N] scores/probabilities per item.
        gts:   [B]    ground-truth class index per row (0..N-1).

    Returns:
        float: mean MRR over the batch.

    Notes:
        For each row, let rank be the 1-based position of the ground-truth item
        in the descending sort of preds. Contribution is 1/rank.
    """
    # --- checks ---
    if not isinstance(preds, (torch.Tensor, np.ndarray)) or not isinstance(gts, (torch.Tensor, np.ndarray)):
        raise TypeError("preds and gts must be torch.Tensor or np.ndarray.")
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D [B, N], got shape {getattr(preds, 'shape', None)}.")
    if gts.ndim != 1 or len(gts) != len(preds):
        raise ValueError("gts must be 1D [B] and have the same length as preds.")

    B, N = int(preds.shape[0]), int(preds.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elem_size = preds.element_size() if isinstance(preds, torch.Tensor) else preds.dtype.itemsize
    rows_per_chunk = max(1, 1_073_741_824 // max(1, N * elem_size))

    rr_list = []
    with torch.no_grad():
        for start in range(0, B, rows_per_chunk):
            end = min(start + rows_per_chunk, B)

            p_slice = preds[start:end]
            g_slice = gts[start:end]

            p_t = p_slice if isinstance(p_slice, torch.Tensor) else torch.as_tensor(p_slice)
            g_t = g_slice if isinstance(g_slice, torch.Tensor) else torch.as_tensor(g_slice)

            p_t = p_t.to(device=device)
            g_t = g_t.to(device=device, dtype=torch.long)

            # gather ground-truth scores: [chunk]
            gt_scores = p_t.gather(dim=1, index=g_t.view(-1, 1)).squeeze(1)

            # rank = 1 + number of items with strictly higher score than gt
            # (ties: items with equal score are NOT counted as higher; gt gets the best possible rank among ties)
            higher = (p_t > gt_scores.unsqueeze(1)).sum(dim=1).to(torch.float32)  # [chunk]
            rank = higher + 1.0

            rr = (1.0 / rank).cpu()
            rr_list.append(rr)

    return torch.cat(rr_list, dim=0).mean().item()


@register_eval("MRR")
def mrr(
    preds: np.ndarray,
    gts: np.ndarray,
) -> float:
    return evaluate_mrr(preds, gts)
