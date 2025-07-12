from torch.utils.data import Dataset
import torch

from typing import List, Dict, Any, Tuple, Callable
import numpy as np
import pandas as pd

def _make_chunks(
    poi: np.ndarray,
    ts: np.ndarray,
    seq_len: int
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
    
    n = len(poi)
    starts = np.arange(0, n - 1, seq_len)
    out: List[Tuple[np.ndarray, ...]] = []

    for s in starts:
        e = s + seq_len
        if e < n:
            out.append((
                poi[s:e],
                ts[s:e],
                np.ones(seq_len, dtype=np.uint8),
                int(poi[e]),
                int(ts[e]),
            ))
        else:
            inp_poi = poi[s:-1]
            inp_ts  = ts[s:-1]
            pad = seq_len - len(inp_poi)
            out.append((
                np.pad(inp_poi, (0, pad), constant_values=0),
                np.pad(inp_ts,  (0, pad), constant_values=0),
                np.pad(np.ones_like(inp_poi, dtype=np.uint8),
                       (0, pad), constant_values=0),
                int(poi[-1]),
                int(ts[-1]),
            ))
    return out


def post_process_func(
    df: pd.DataFrame,
    sequence_length: int,
    n_jobs: int = 1,
    backend: str = "loky"
) -> List[Dict[str, Any]]:
    
    df = df.sort_values("timestamps", kind="mergesort")

    group_key = ['user_id', 'trajectory_id'] \
        if 'trajectory_id' in df.columns else ['user_id']
    grouped = list(df.groupby(group_key, sort=False))

    def _worker(name, g):
        uid = name[0] if isinstance(name, tuple) else int(name)
        poi = g['POI_id'].to_numpy(np.int64,  copy=False)
        ts  = g['timestamps'].to_numpy(np.int64, copy=False)
        samples = _make_chunks(poi, ts, sequence_length)
        return [
            {
                'user_id': uid,
                'POI_id': s[0],
                'timestamps': s[1],
                'mask': s[2],
                'y_POI_id': s[3],
                'y_timestamp': s[4],
            } for s in samples
        ]

    if n_jobs == 1:
        data_nested = [_worker(name, g) for name, g in grouped]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as err:
            raise ImportError(
                "please install joblib to use parallel processing"
            ) from err
        data_nested = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_worker)(name, g) for name, g in grouped
        )

    return [item for sub in data_nested for item in sub]

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
    ):
        super().__init__()
        raw_df = preprocess_func()
        self.samples = post_process_func(
            raw_df, sequence_length=sequence_length, n_jobs=n_jobs
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {k: torch.as_tensor(v) for k, v in sample.items()}