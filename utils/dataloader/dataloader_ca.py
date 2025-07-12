from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader, default_collate

from traj_lib.utils.register import register_dataloader
from traj_lib.utils.logger import get_logger
from traj_lib.utils.dataloader.dataloader_base import BaseDataset
from traj_lib.utils.exargs import ConfigResolver

logger = get_logger(__name__)
dataset_name = 'CA'
root_path = Path(__file__).resolve().parent.parent.parent
args = ConfigResolver(f'{root_path}/data/{dataset_name}/{dataset_name}.yaml').parse()


def pre_process_func() -> pd.DataFrame:
    df = pd.read_csv(f'{root_path}/data/{dataset_name}/{dataset_name}_{args.get("split", "train")}.csv')
    df = df[[
        "UTCTimeOffsetEpoch",
        "trajectory_id",
        "POI_id",
        "user_id",
    ]]
    df["POI_id"] = pd.factorize(df["POI_id"])[0] + 1
    df["user_id"] = pd.factorize(df["user_id"])[0] + 1
    df["timestamps"] = pd.to_datetime(
        df["UTCTimeOffsetEpoch"], unit="ns", errors="coerce"
    ).view("int64")
    df = df.drop(columns=["UTCTimeOffsetEpoch"])
    df = df[["timestamps", "trajectory_id", "POI_id", "user_id"]]
    df = df.sort_values("timestamps")
    return df

class MyDataset(BaseDataset):
    def __init__(self) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            sequence_length=args.get("sequence_length", 30),
            n_jobs=args.get("n_jobs", 1)
        )

@register_dataloader(name=dataset_name)
class MyDataLoader(DataLoader):
    def __init__(self) -> None:
        super().__init__(
            dataset=MyDataset(),
            batch_size=args.get("batch_size", 32),
            collate_fn=default_collate
        )
        logger.info("DataLoader initialized with dataset: %s", dataset_name)
        logger.info("Batch size: %d, Total Batches: %d",
                    args.get("batch_size", 32), len(self.dataset) // args.get("batch_size", 32))
        logger.info("Sequence length: %d", args.get("sequence_length", 30))
        logger.info("Number of jobs for preprocessing: %d", args.get("n_jobs", 1))