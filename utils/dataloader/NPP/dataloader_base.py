from typing import List, Dict, Any, Callable, Protocol, Generic, TypeVar, cast


import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from pandas.core.groupby.generic import DataFrameGroupBy  # 需要 pandas-stubs


from utils.register import VIEW_REGISTRY
from collections import OrderedDict


def build_samples_same_schema(
    groupby_user: DataFrameGroupBy, sequence_length: int, timestamps_col: str = "timestamps"
) -> List[Dict[str, Any]]:
    data_list: List[Dict[str, Any]] = []

    for index, data in groupby_user:
        user_id = index[0] if isinstance(index, tuple) else np.int64(index)

        data = data.sort_values(by=timestamps_col, kind="mergesort")

        cols = list(data.columns)

        col_arrays = {c: data[c].to_numpy(copy=False) for c in cols}
        n = len(data)
        if n < 2:
            continue

        start_idx = 0
        traj_base = len(data_list)

        pad_cache_numeric : Dict[str, np.ndarray] = {}
        pad_cache_object = None

        while start_idx < n - 1:
            end_idx = start_idx + sequence_length
            slice_end = min(end_idx, n - 1)
            real_len = slice_end - start_idx
            pad_num = sequence_length - real_len

            y_dict: Dict[str, Any] = {c: col_arrays[c][slice_end] for c in cols}
            sample: Dict[str, Any] = {  
                "user_id": user_id,
                "mask": int(real_len),
                "trajectory_id": len(data_list),
                "y_POI_id": y_dict,
            }
            sample["y_POI_id"]["trajectory_id"] = sample["trajectory_id"]

            for c in cols:
                if c in sample:
                    continue

                seq = col_arrays[c][start_idx:slice_end]

                is_num = np.issubdtype(seq.dtype, np.number)

                if pad_num > 0:
                    if is_num:
                        key = seq.dtype.str
                        tmpl = pad_cache_numeric.get(key)
                        if tmpl is None or tmpl.shape[0] != sequence_length:
                            tmpl = np.zeros(sequence_length, dtype=seq.dtype)
                            pad_cache_numeric[key] = tmpl
                        out = tmpl.copy()
                        out[:real_len] = seq
                    else:
                        if (pad_cache_object is None) or (
                            pad_cache_object.shape[0] != sequence_length
                        ):
                            pad_cache_object = np.empty(sequence_length, dtype=object)
                            pad_cache_object[:] = None
                        out = pad_cache_object.copy()
                        out[:real_len] = seq
                else:
                    out = seq.copy()

                sample[c] = out

            data_list.append(sample)
            start_idx += sequence_length

    return data_list


def post_process_func(
    df: pd.DataFrame, sequence_length: int, backend: str = "loky"
) -> List[Dict[str, Any]]:
    df = df.sort_values(by="timestamps")
    group_key = (
        ["user_id", "trajectory_id"] if "trajectory_id" in df.columns else ["user_id"]
    )
    groupby_user = df.groupby(group_key)
    data_list = build_samples_same_schema(
        groupby_user, sequence_length, timestamps_col="timestamps"
    )

    return data_list


def flex_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Recursively do the following for a batch (list[dict]):
      - If all values are Tensors → stack
      - If all values are numbers → tensor
      - Otherwise collect into a list (keep order)

    Allow different samples to have different keys: fill in None for missing keys.
    """
    # (1) First find the 'union key' of all samples
    all_keys = set().union(*batch)
    collated: Dict[str, Any] = {}
    for k in all_keys:
        values = [b.get(k, None) for b in batch]

        # ========= Triple recursive judgement =========
        # (a) All Tensors with consistent shape
        if all(isinstance(v, torch.Tensor) for v in values):
            tensors = cast(List[torch.Tensor], values)
            collated[k] = torch.stack(tensors)
        # (b) All numerical values
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            collated[k] = torch.tensor(values)
        # (c) All dict → recursion
        elif all(isinstance(v, dict) for v in values if v is not None):
            # Note: None is allowed; Replace None with an empty dict recursively
            sub_batch = [{**(v or {})} for v in values]
            collated[k] = flex_collate(sub_batch)
        else:
            # Other types (str, list, mixed) → Original list
            collated[k] = values

    return collated


def _maybe_tensor(x: Any) -> Any:
    """Numerical value → Tensor; the rest remains unchanged"""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
        return torch.as_tensor(x)
    return x


class BaseDataset(Dataset):
    def __init__(
        self,
        preprocess_func: Callable[..., pd.DataFrame],
        sequence_length: int,
        n_jobs: int = 1,
        pre_views: List[str] | None = None,
        post_views: List[str] | None = None,
    ) -> None:
        super().__init__()
        raw_df_pre = preprocess_func()
        view_value: Dict[str, Any] = {}
        if pre_views:
            for view in pre_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df_pre, view_value = VIEW_REGISTRY[view](raw_df_pre, view_value)

        raw_df_post = post_process_func(raw_df_pre, sequence_length=sequence_length)
        if post_views:
            for view in post_views:
                if view not in VIEW_REGISTRY:
                    raise ValueError(f"View '{view}' not registered")
                raw_df_post, view_value = VIEW_REGISTRY[view](raw_df_post, view_value)
        self.samples = raw_df_post
        self.view_value = view_value

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        # Need to maintain insertion order with OrderedDict available; Otherwise the regular dict will also maintain its order
        ordered = OrderedDict()
        for k, v in s.items():
            ordered[k] = _maybe_tensor(v)
        return ordered


T = TypeVar("T", bound=BaseDataset)
T_co = TypeVar("T_co", bound="BaseDataset", covariant=True)


class DatasetCtor(Protocol[T_co]):
    def __call__(
        self,
        *,
        pre_views: List[str] | None = None,
        post_views: List[str] | None = None,
    ) -> T_co: ...


class BaseDataLoader(Generic[T]):
    def __init__(
        self,
        MyDataset: DatasetCtor[T],
        dataset_name: str,
        logger: Any,
        args: Dict[str, Any],
        model_args: Dict[str, Any],
        pre_views: List[str] | None = None,
        post_views: List[str] | None = None,
    ) -> None:
        dataset = MyDataset(pre_views=pre_views, post_views=post_views)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.dataset_name = dataset_name
        
        torch.manual_seed(args.get("seed", 42))
        np.random.seed(args.get("seed", 42))
        train_dataset, test_dataset, val_dataset = random_split(
            dataset, lengths=[train_size, val_size, test_size]
        )

        train_batch_size = model_args.get(
            "train_batch_size", model_args.get("batch_size", 32)
        )
        val_batch_size = model_args.get(
            "val_batch_size", model_args.get("batch_size", 32)
        )
        test_batch_size = model_args.get(
            "test_batch_size", model_args.get("batch_size", 32)
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=train_batch_size, collate_fn=flex_collate
        )
        self.val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=val_batch_size, collate_fn=flex_collate
        )
        self.test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=test_batch_size, collate_fn=flex_collate
        )

        self.view_value = dataset.view_value
        logger.info("DataLoader initialized with dataset: %s", dataset_name)
        logger.info("Total samples: %d", len(dataset))
        logger.info(
            "Train Batch size: %d, Total Batches: %d",
            train_batch_size,
            len(self.train_dataloader) // train_batch_size,
        )
        logger.info(
            "Validation Batch size: %d, Total Batches: %d",
            val_batch_size,
            len(self.val_dataloader) // val_batch_size,
        )
        logger.info(
            "Test Batch size: %d, Total Batches: %d",
            test_batch_size,
            len(self.test_dataloader) // test_batch_size,
        )
        logger.info("Sequence length: %d", args.get("sequence_length", 30))
        logger.info("Number of jobs for preprocessing: %d", args.get("n_jobs", 1))
