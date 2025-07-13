from torch.utils.data import Dataset
import torch

from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import pandas as pd

from traj_lib.utils.register import VIEW_REGISTRY

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # 可选并行
from torch.utils.data import Dataset

def _make_chunks_generic(
    arrays: Dict[str, np.ndarray],
    seq_len: int,
    pad_value: int = 0
) -> List[Dict[str, Any]]:
    """
    arrays: {col_name: 1D np.ndarray of length n}
    返回每一个滑窗样本，dict 包含：
      - 对每个 col_name: 长度为 seq_len 的 np.ndarray
      - 对应 y_col_name: 下一个时刻标量
      - mask: np.ndarray[seq_len]，1 表示真实，0 表示 padding
    """
    n = next(iter(arrays.values())).shape[0]
    starts = np.arange(0, n - 1, seq_len)
    samples: List[Dict[str, Any]] = []
    for s in starts:
        e = s + seq_len
        is_full = (e < n)
        mask = np.ones(seq_len, dtype=bool)
        if not is_full:
            mask[: n - 1 - s] = True
            mask[n - 1 - s :] = False

        sample: Dict[str, Any] = {}
        for col, arr in arrays.items():
            if is_full:
                sample[col] = arr[s:e]
                sample[f"y_{col}"] = arr[e]
            else:
                inp = arr[s : n - 1]
                pad = seq_len - inp.shape[0]
                sample[col] = np.pad(inp, (0, pad), constant_values=pad_value)
                sample[f"y_{col}"] = arr[-1]
        sample["mask"] = mask.astype(np.uint8)
        samples.append(sample)
    return samples


def post_process_func(
    df: pd.DataFrame,
    sequence_length: int,
    n_jobs: int = 1,
    backend: str = "loky"
) -> List[Dict[str, Any]]:
    # 1. 排序
    df = df.sort_values("timestamps", kind="mergesort").reset_index(drop=True)

    # 2. 分组键
    group_key = (
        ["user_id", "trajectory_id"]
        if "trajectory_id" in df.columns
        else ["user_id"]
    )
    grouped = list(df.groupby(group_key, sort=False))

    # 3. 需要滑窗的列：除分组键之外的所有列
    feature_cols = [c for c in df.columns if c not in group_key]

    def _worker(name: Tuple, g: pd.DataFrame):
        # 提取 user_id, trajectory_id
        out: List[Dict[str, Any]] = []
        uid = name[0] if isinstance(name, tuple) else int(name)

        # 构造每列的 np.ndarray
        arrays = {
            col: g[col].to_numpy(copy=False) for col in feature_cols
        }

        # 生成切片样本
        slices = _make_chunks_generic(arrays, sequence_length)
        # 每个样本中都加上 user_id 和 trajectory_id（如果有）
        for s in slices:
            s["user_id"] = uid
            if isinstance(name, tuple):
                s["trajectory_id"] = name[1]
            out.append(s)
        return out

    if n_jobs == 1:
        nested = [_worker(name, grp) for name, grp in grouped]
    else:
        nested = Parallel(
            n_jobs=n_jobs, backend=backend
        )(delayed(_worker)(name, grp) for name, grp in grouped)

    # flatten
    return [sample for sub in nested for sample in sub]


def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    collated = {}
    for k in batch[0]:
        if isinstance(batch[0][k], np.ndarray):
            collated[k] = torch.as_tensor(
                [b[k] for b in batch], dtype=torch.long
            )
        else:
            collated[k] = torch.as_tensor(
                [b[k] for b in batch]
            )
    return collated

class BaseDataset(Dataset):
    def __init__(
        self,
        preprocess_func: Callable[..., pd.DataFrame],
        sequence_length: int,
        n_jobs: int = 1,
        pre_views: List[str] = None,
        post_views: List[str] = None
    ) -> None:
        super().__init__()
        raw_df = preprocess_func()
        view_value = {}
        if pre_views:
            for view in pre_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
        raw_df = post_process_func(
            raw_df, sequence_length=sequence_length, n_jobs=n_jobs
        )
        if post_views:
            for view in post_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df, view_value = VIEW_REGISTRY[view](raw_df, view_value)
        self.samples = raw_df
        self.view_value = view_value

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {k: torch.as_tensor(v) for k, v in sample.items()}