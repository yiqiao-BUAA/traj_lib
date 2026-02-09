# ./model/STGCN/main.py
from __future__ import annotations

from typing import Dict, Any, Iterable, Callable, Tuple, List
import inspect
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from utils.earlystopping import EarlyStopping
from utils.GPU_find import find_gpu

from .STGCN import CandidateSubgraphTFModel


# ---------------------------------------------------------------------
# Config & Global Setup
# ---------------------------------------------------------------------
model_args = ConfigResolver("./model/STGCN/STGCN.yaml").parse()
log = get_logger(__name__)

# 你现在 pipeline 需要这些 view
pre_views: list[str] = ["common_count", "adj_view", "subgraph_view"]
post_views: list[str] = ["seq_len"]

_device: torch.device = find_gpu()


model: nn.Module | None = None

_A_global_cpu: torch.Tensor | None = None
_num_nodes: int | None = None


# =====================================================================
# Tensor helpers
# =====================================================================
def _ensure_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def _infer_lengths(batch: Dict[str, Any]) -> torch.Tensor:
    """
    Returns [B] int lengths.
    Supports:
      - batch["mask"] as [B] lengths
      - batch["mask"] as [B,S] mask
      - else POI_id != 0
    """
    poi = _ensure_tensor(batch["POI_id"]).long()
    B, S = poi.shape

    if "mask" in batch:
        m = _ensure_tensor(batch["mask"])
        if m.dim() == 1:
            return m.long().clamp(min=0, max=S)
        if m.dim() == 2:
            if m.dtype == torch.bool:
                return m.long().sum(-1).clamp(min=0, max=S)
            return m.ne(0).long().sum(-1).clamp(min=0, max=S)

    return poi.ne(0).long().sum(-1).clamp(min=0, max=S)


def _lengths_to_mask(lengths: torch.Tensor, S: int, device: torch.device) -> torch.Tensor:
    """
    lengths: [B] int
    returns: [B,S] bool (positions < length are True)
    """
    idx = torch.arange(S, device=device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).bool()


def _infer_seq_mask(batch: Dict[str, Any]) -> torch.Tensor:
    """
    Returns [B,S] bool mask
    """
    poi = _ensure_tensor(batch["POI_id"]).long()
    B, S = poi.shape

    if "mask" in batch:
        m = _ensure_tensor(batch["mask"])
        if m.dim() == 2:
            return m.bool() if m.dtype == torch.bool else m.ne(0)
        if m.dim() == 1:
            lens = m.long().clamp(min=0, max=S)
            return _lengths_to_mask(lens, S, poi.device)

    return poi.ne(0)


def _gather_labels(batch: Dict[str, Any]) -> torch.Tensor:
    """
    兼容你原来的 label 结构（单步 next-POI），但本 pipeline 用的是 teacher forcing，
    训练时标签来自 x[:,1:]，这里只在某些评估/兼容场景会用到。
    """
    if "y_POI_id" in batch and isinstance(batch["y_POI_id"], dict) and "POI_id" in batch["y_POI_id"]:
        return _ensure_tensor(batch["y_POI_id"]["POI_id"]).long()
    if "y" in batch:
        return _ensure_tensor(batch["y"]).long()
    raise KeyError("Label not found in batch: expected batch['y_POI_id']['POI_id'] or batch['y'].")


# =====================================================================
# Global A init (CPU-only)
# =====================================================================
def _extract_adj(view_value: Dict[str, Any], num_nodes: int) -> torch.Tensor:
    """
    Extract dense adjacency A from view_value['adj'].
    Supports:
      - adj is 2D array/tensor
      - adj is dict with keys in ("adj","A","graph")
    """
    A = view_value.get("adj", None)
    if isinstance(A, dict):
        for k in ("adj", "A", "graph"):
            if k in A:
                A = A[k]
                break

    if A is None:
        raise ValueError("view_value['adj'] is missing.")

    A = _ensure_tensor(A).float()
    if A.dim() != 2:
        raise ValueError(f"adj must be 2D, got shape={tuple(A.shape)}")

    # pad/trim to [N,N]
    n0 = A.size(0)
    if n0 != num_nodes:
        if n0 < num_nodes:
            pad = num_nodes - n0
            A = F.pad(A, (0, pad, 0, pad))
        else:
            A = A[:num_nodes, :num_nodes]

    return A.clamp(min=0)


def _maybe_init_global_A(view_value: Dict[str, Any]) -> None:
    global _A_global_cpu, _num_nodes
    if _A_global_cpu is not None:
        return

    _num_nodes = int(view_value["n_item"])
    A = _extract_adj(view_value, _num_nodes).cpu().contiguous()
    _A_global_cpu = A
    log.info(f"[SubGraphTF] Global dense A cached on CPU: shape={tuple(A.shape)}")


def _slice_A_sub_for_nodes(sub_nodes: torch.Tensor) -> torch.Tensor:
    """
    sub_nodes: [n_sub] global ids (CPU tensor preferred)
    returns:   [n_sub,n_sub] dense sub adjacency on CPU
    """
    assert _A_global_cpu is not None
    idx = sub_nodes.long()
    return _A_global_cpu.index_select(0, idx).index_select(1, idx)





# =====================================================================
# Subgraph construction for teacher forcing (group by prev_id)
# =====================================================================
def _get_sub_candidates(view_value: Dict[str, Any], prev_id: int) -> List[int]:
    """
    view_value['sub_graph'] keys assumed str(prev_id).
    """
    sg = view_value["subgraphs"]
    cand = sg.get(str(prev_id), None)
    if cand is None:
        return []
    # ensure python list[int]
    return [int(x) for x in cand]


def _build_candidates_for_key(
    prev_id: int,
    labels_for_key: List[int],
    view_value: Dict[str, Any],
    num_nodes: int,
    subgraph_size: int,
) -> List[int]:
    """
    Build a fixed-size candidate list for a given prev_id, shared across all positions in the batch
    that have this prev_id.

    Strategy:
      - include prev_id itself (if valid)
      - include the most frequent labels for this key (so all positions can be supervised)
      - fill remaining from view_value['sub_graph'][prev_id]
      - pad with 0 if still short
    """
    nodes: List[int] = []

    # include prev_id (self)
    if 0 < prev_id < num_nodes:
        nodes.append(prev_id)

    # include labels (already frequency-sorted upstream)
    for y in labels_for_key:
        if y == 0 or y < 0 or y >= num_nodes:
            continue
        if y not in nodes:
            nodes.append(y)
        if len(nodes) >= subgraph_size:
            break

    # fill by distance-sampled neighbors
    for nid in _get_sub_candidates(view_value, prev_id):
        if nid == 0 or nid < 0 or nid >= num_nodes:
            continue
        if nid not in nodes:
            nodes.append(nid)
        if len(nodes) >= subgraph_size:
            break

    # pad
    if len(nodes) < subgraph_size:
        nodes.extend([0] * (subgraph_size - len(nodes)))
    else:
        nodes = nodes[:subgraph_size]

    return nodes


def _group_positions_by_prev_id(
    prev_ids: torch.Tensor,  # [Npos] on GPU
    y_global: torch.Tensor,  # [Npos] on GPU
) -> Dict[int, Dict[str, Any]]:
    """
    Group flattened positions by prev_id, also collect label frequency per key.
    Returns dict[key] -> {"pos_idx": Tensor[...], "label_freq": Dict[int,int]}
    """
    prev_cpu = prev_ids.detach().cpu().numpy().tolist()
    y_cpu = y_global.detach().cpu().numpy().tolist()

    groups: Dict[int, Dict[str, Any]] = {}
    for i, (p, y) in enumerate(zip(prev_cpu, y_cpu)):
        p = int(p)
        y = int(y)
        if p not in groups:
            groups[p] = {"pos_idx": [], "label_freq": {}}
        groups[p]["pos_idx"].append(i)
        if y != 0:
            groups[p]["label_freq"][y] = groups[p]["label_freq"].get(y, 0) + 1

    # convert pos_idx to tensors later in forward (we keep list now)
    return groups


def _sorted_labels_by_freq(label_freq: Dict[int, int], limit: int) -> List[int]:
    """
    Return labels sorted by descending frequency.
    """
    items = sorted(label_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in items[:limit]]


# =====================================================================
# Core forward for teacher forcing
# =====================================================================
@torch.no_grad()
def _estimate_num_valid_tokens(mask: torch.Tensor) -> int:
    return int(mask.long().sum().detach().cpu().item())


def _forward_teacher_forcing(
    batch: Dict[str, Any],
    view_value: Dict[str, Any],
    subgraph_size: int,
) -> Dict[str, Any]:
    """
    Returns a dict containing flattened token-level predictions for teacher forcing:
      logits : [Npos, subgraph_size] (GPU)
      y_local: [Npos] (GPU) in [0, subgraph_size-1] or -100 ignore
      cand   : [Npos, subgraph_size] global ids (GPU)
      gts    : [Npos] global ids (GPU)
    """
    assert model is not None
    assert _num_nodes is not None

    x_full = _ensure_tensor(batch["POI_id"]).long().to(_device)     # [B,S]
    lengths = _infer_lengths(batch).to(_device)                     # [B]
    B, S = x_full.shape
    if S < 2:
        # no valid next-token supervision
        empty = torch.empty((0, subgraph_size), device=_device)
        return {"logits": empty, "y_local": torch.empty((0,), dtype=torch.long, device=_device),
                "cand": empty.long(), "gts": torch.empty((0,), dtype=torch.long, device=_device)}

    # teacher forcing shift
    x_in = x_full[:, :-1]                 # [B,S-1]
    y_tgt = x_full[:, 1:]                 # [B,S-1]

    # valid prediction steps: t < lengths-1
    lengths_pred = (lengths - 1).clamp(min=0, max=S - 1)
    mask_in = _lengths_to_mask(lengths_pred, S - 1, _device)        # [B,S-1] bool

    # additionally ignore positions where prev or y is padding(0)
    mask_in = mask_in & x_in.ne(0) & y_tgt.ne(0)
    if not mask_in.any():
        empty = torch.empty((0, subgraph_size), device=_device)
        return {"logits": empty, "y_local": torch.empty((0,), dtype=torch.long, device=_device),
                "cand": empty.long(), "gts": torch.empty((0,), dtype=torch.long, device=_device)}

    # encode per-step hidden states
    assert isinstance(model, CandidateSubgraphTFModel)
    h_seq = model.encode(x_in, mask_in)                # [B,S-1,D]

    # flatten valid positions
    h_flat = h_seq[mask_in]                            # [Npos,D]
    prev_flat = x_in[mask_in]                          # [Npos]
    y_flat = y_tgt[mask_in]                            # [Npos]

    Npos = h_flat.size(0)
    logits_flat = torch.zeros((Npos, subgraph_size), device=_device, dtype=torch.float32)
    cand_flat = torch.zeros((Npos, subgraph_size), device=_device, dtype=torch.long)
    y_local_flat = torch.full((Npos,), -100, device=_device, dtype=torch.long)

    # group positions by prev_id to share candidate set per key within batch
    groups = _group_positions_by_prev_id(prev_flat, y_flat)

    # build per-key candidates and score
    for key_prev, info in groups.items():
        if key_prev == 0:
            continue

        pos_list: List[int] = info["pos_idx"]
        pos_idx = torch.tensor(pos_list, device=_device, dtype=torch.long)

        # frequency-sorted labels for this key, limited to avoid exceeding subgraph_size
        # reserve space: at least 1 for key_prev (if valid), so labels limit = subgraph_size-1
        labels_sorted = _sorted_labels_by_freq(info["label_freq"], limit=max(0, subgraph_size - 1))

        nodes = _build_candidates_for_key(
            prev_id=int(key_prev),
            labels_for_key=labels_sorted,
            view_value=view_value,
            num_nodes=int(_num_nodes),
            subgraph_size=subgraph_size,
        )

        sub_nodes = torch.tensor(nodes, device=_device, dtype=torch.long)         # [n_sub]
        # slice A_sub on CPU then move to GPU
        A_sub_cpu = _slice_A_sub_for_nodes(sub_nodes.detach().cpu())              # [n_sub,n_sub] CPU
        A_sub = A_sub_cpu.to(_device, non_blocking=True)                          # GPU

        # candidate embeddings
        cand_emb = model.candidate_embed(sub_nodes, A_sub)                        # [n_sub,D]

        # logits for this key group
        h_key = h_flat.index_select(0, pos_idx)                                   # [n_pos,D]
        logits_key = h_key @ cand_emb.transpose(0, 1)                             # [n_pos,n_sub]

        logits_flat.index_copy_(0, pos_idx, logits_key)
        cand_flat.index_copy_(0, pos_idx, sub_nodes.unsqueeze(0).expand(pos_idx.size(0), -1))

        # map y_global to y_local within this candidate list
        # build mapping on CPU (n_sub small)
        g2l = {int(nid): j for j, nid in enumerate(nodes) if int(nid) != 0}
        y_key = y_flat.index_select(0, pos_idx).detach().cpu().numpy().tolist()
        y_local_list = [g2l.get(int(y), -100) for y in y_key]
        y_local = torch.tensor(y_local_list, device=_device, dtype=torch.long)
        y_local_flat.index_copy_(0, pos_idx, y_local)

    return {"logits": logits_flat, "y_local": y_local_flat, "cand": cand_flat, "gts": y_flat}


# =====================================================================
# Init model
# =====================================================================
def _maybe_init_model(view_value: Dict[str, Any]) -> None:
    global model
    if model is not None:
        return

    _maybe_init_global_A(view_value)

    vocab_size = int(view_value["n_item"])
    d_model = int(model_args.get("d_model", 128))
    dropout = float(model_args.get("dropout", 0.1))
    tcn_kernel = int(model_args.get("tcn_kernel", 3))
    gcn_layers = int(model_args.get("gcn_layers", 1))

    model = CandidateSubgraphTFModel(
        vocab_size=vocab_size,
        d_model=d_model,
        dropout=dropout,
        tcn_kernel=tcn_kernel,
        gcn_layers=gcn_layers,
    ).to(_device)

    log.info(
        f"[SubGraphTF] Model initialized: vocab={vocab_size}, d_model={d_model}, "
        f"subgraph_size={int(model_args.get('subgraph_size', 128))}, "
        f"tcn_kernel={tcn_kernel}, gcn_layers={gcn_layers}"
    )


# =====================================================================
# Train / Inference
# =====================================================================
def _call_eval_func(func: Callable, pred: np.ndarray, gts: np.ndarray, cand: np.ndarray | None = None):
    """
    Flexible calling:
      - if func expects 3 args -> func(pred, cand, gts)
      - elif expects 2 args -> try func(pred, gts)
      - else: raise
    """
    sig = inspect.signature(func)
    n_params = len(sig.parameters)
    if n_params >= 3:
        if cand is None:
            raise ValueError("eval func expects cand but cand is None.")
        return func(pred, cand, gts)
    if n_params == 2:
        return func(pred, gts)
    raise ValueError("Unsupported eval func signature.")


def _train_impl(
    train_dataloader: Any,
    val_dataloader: Any,
    view_value: Dict[str, Any],
    eval_funcs: Dict[str, Callable],
    **cfg: Any,
) -> Iterable[list[dict[str, Any]]]:
    _maybe_init_model(view_value)
    assert model is not None
    assert _num_nodes is not None

    subgraph_size = int(model_args.get("subgraph_size", 128))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_args["lr"]),
        weight_decay=float(model_args["weight_decay"]),
    )

    # Early stopping (可选)
    monitor_metric = next(iter(eval_funcs.keys())) if eval_funcs else None
    early_stopper = None
    higher_is_better = False
    if monitor_metric is not None:
        early_stopper = EarlyStopping(
            patience=int(model_args.get("patience", 5)),
            save_model_folder="./saved_models",
            save_model_name="STGCN_SubGraphTF_best",
            logger=log,
        )
        higher_is_better = "acc" in monitor_metric.lower() or "recall" in monitor_metric.lower()

    for ep in range(1, int(model_args["epochs"]) + 1):
        model.train()
        running_loss, n_tok = 0.0, 0

        for batch in train_dataloader:
            out = _forward_teacher_forcing(batch, view_value, subgraph_size=subgraph_size)
            logits = out["logits"]        # [Npos,n_sub]
            y_local = out["y_local"]      # [Npos]

            if logits.numel() == 0:
                continue

            loss = F.cross_entropy(logits, y_local, ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # token-avg loss
            valid = y_local.ne(-100)
            denom = int(valid.long().sum().detach().cpu().item())
            denom = max(1, denom)
            running_loss += float(loss.detach().cpu().item()) * denom
            n_tok += denom

        avg_train_loss = running_loss / max(1, n_tok)
        log.info(f"[SubGraphTF][Epoch {ep}/{model_args['epochs']}] Train Loss(tok): {avg_train_loss:.6f}")

        # validation
        inference_res = inference(val_dataloader, view_value, **cfg)
        scores = {}
        if eval_funcs:
            pred = inference_res["pred"]
            gts = inference_res["gts"]
            cand = inference_res.get("cand", None)
            for name, func in eval_funcs.items():
                try:
                    scores[name] = _call_eval_func(func, pred, gts, cand=cand)
                except TypeError:
                    # fallback: if user eval expects predicted ids, convert top1 global
                    if cand is not None:
                        top1_idx = np.argmax(pred, axis=1)
                        top1_global = cand[np.arange(cand.shape[0]), top1_idx]
                        scores[name] = func(top1_global, gts)
                    else:
                        raise

            log.info(f"[SubGraphTF][Epoch {ep}] Validation Scores: {scores}")

        yield [scores, {"train_loss": avg_train_loss}]

        if early_stopper is not None and monitor_metric in scores:
            if early_stopper.step([(monitor_metric, scores[monitor_metric], higher_is_better)], model):
                log.info(f"[SubGraphTF] Early stopping at epoch {ep}.")
                early_stopper.load_checkpoint(model)
                break


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    view_value: Dict[str, Any],
    eval_funcs: Dict[str, Callable],
    **kwargs,
) -> Iterable[list[dict[str, Any]]]:
    return _train_impl(train_dataloader, val_dataloader, view_value, eval_funcs, **kwargs)


@torch.no_grad()
def inference(
    test_dataloader: DataLoader,
    view_value: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    assert view_value is not None, "view_value is required (needs n_item/adj_view/sub_graph)."
    _maybe_init_model(view_value)
    assert model is not None

    subgraph_size = int(model_args.get("subgraph_size", 128))

    model.eval()
    all_logits: list[torch.Tensor] = []
    all_gts: list[torch.Tensor] = []
    all_cand: list[torch.Tensor] = []

    for batch in test_dataloader:
        out = _forward_teacher_forcing(batch, view_value, subgraph_size=subgraph_size)
        logits = out["logits"]
        y_local = out["y_local"]
        cand = out["cand"]
        gts = out["gts"]

        if logits.numel() == 0:
            continue

        # keep only supervised positions (optional; makes metrics cleaner)
        keep = y_local.ne(-100)
        if keep.any():
            all_logits.append(logits[keep].detach().cpu())
            all_cand.append(cand[keep].detach().cpu())
            all_gts.append(gts[keep].detach().cpu())

    if len(all_logits) == 0:
        return {
            "pred": np.zeros((0, subgraph_size), dtype=np.float32),
            "cand": np.zeros((0, subgraph_size), dtype=np.int64),
            "gts": np.zeros((0,), dtype=np.int64),
        }

    pred = torch.cat(all_logits, dim=0).numpy()   # [Npos,n_sub]
    cand = torch.cat(all_cand, dim=0).numpy()     # [Npos,n_sub] global ids
    gts = torch.cat(all_gts, dim=0).numpy()       # [Npos] global ids

    return {"pred": pred, "cand": cand, "gts": gts}
