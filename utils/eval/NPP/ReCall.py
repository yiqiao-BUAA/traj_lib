from tqdm import tqdm
from typing import Union

import numpy as np
import torch

from utils.register import register_eval
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(
    preds: Union[torch.Tensor, np.ndarray],
    gts:   Union[torch.Tensor, np.ndarray],
    topk: int = 20
) -> float:
    
    if not isinstance(preds, (torch.Tensor, np.ndarray)) or not isinstance(gts, (torch.Tensor, np.ndarray)):
        raise TypeError("preds and gts must be torch.Tensor or np.ndarray.")
    if preds.ndim != 2:
        raise ValueError(f"preds should be 2D [B, N], got shape {getattr(preds, 'shape', None)}.")
    if gts.ndim != 1 or len(gts) != len(preds):
        raise ValueError("gts must be 1D [B] and have the same length as preds.")
    if topk <= 0:
        raise ValueError("topk must be a positive integer.")

    B, N = int(preds.shape[0]), int(preds.shape[1])
    K = int(min(topk, N))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elem_size = preds.element_size() if isinstance(preds, torch.Tensor) else preds.dtype.itemsize
    rows_per_chunk = max(1, 1_073_741_824 // max(1, N * elem_size))

    recalls = []
    with torch.no_grad():
        # for start in tqdm(range(0, B, rows_per_chunk), desc=f"Evaluating ReCall {topk:>2}", unit="chunk"):
        for start in range(0, B, rows_per_chunk):  # disable tqdm for cleaner logging
            end = min(start + rows_per_chunk, B)

            p_slice = preds[start:end]
            g_slice = gts[start:end]

            p_t = p_slice if isinstance(p_slice, torch.Tensor) else torch.as_tensor(p_slice)
            g_t = g_slice if isinstance(g_slice, torch.Tensor) else torch.as_tensor(g_slice)

            p_t = p_t.to(device=device)
            g_t = g_t.to(device=device, dtype=torch.long)

            topk_idx = p_t.topk(k=K, dim=1).indices            # [chunk, K]
            hit_any  = (topk_idx == g_t.unsqueeze(1)).any(1)   # [chunk] bool
            recalls.append(hit_any.float().cpu())

    return torch.cat(recalls, 0).mean().item()



@register_eval("ReCall1")
def recall1(
    preds: torch.Tensor | np.ndarray,
    gts: torch.Tensor | np.ndarray,
    topk: int=1
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("ReCall5")
def recall5(
    preds: torch.Tensor | np.ndarray,
    gts: torch.Tensor | np.ndarray,
    topk: int=5
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("ReCall10")
def recall10(
    preds: torch.Tensor | np.ndarray,
    gts: torch.Tensor | np.ndarray,
    topk: int=10
) -> float:
    return evaluate(preds, gts, topk=topk)


@register_eval("ReCall20")
def recall20(
    preds: torch.Tensor | np.ndarray,
    gts: torch.Tensor | np.ndarray,
    topk: int=20
) -> float:
    return evaluate(preds, gts, topk=topk)
