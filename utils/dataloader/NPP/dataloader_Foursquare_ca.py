from pathlib import Path

import pandas as pd

from utils.register import register_dataloader
from utils.logger import get_logger
from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
from utils.exargs import ConfigResolver

logger = get_logger(__name__)
dataset_name = "Foursquare_ca"
root_path = Path(__file__).resolve().parent.parent.parent.parent
args = ConfigResolver(f"{root_path}/data/Foursquare/{dataset_name}.yaml").parse()


def pre_process_func() -> pd.DataFrame:
    df = pd.read_csv(f"{root_path}/data/Foursquare/{dataset_name}.csv")
    df = df[["timestamps", "POI_id", "user_id", "latitude", "longitude", "POI_catid"]]

    df["POI_id"] = pd.factorize(df["POI_id"])[0] + 1
    df["user_id"] = pd.factorize(df["user_id"])[0] + 1
    df["POI_catid"] = pd.factorize(df["POI_catid"])[0] + 1
    df["timestamps"] = pd.to_datetime(
        df["timestamps"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce"
    )  # Processing time(contains damaged data), better not modify
    df = df.dropna(axis=0, how="any")
    df["timestamps"] = df["timestamps"].astype("int64") // 10**9
    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


class MyDataset(BaseDataset):
    def __init__(self, pre_views=None, post_views=None) -> None:
        super().__init__(
            preprocess_func=pre_process_func,
            sequence_length=args.get("sequence_length", 30),
            n_jobs=args.get("n_jobs", 1),
            pre_views=pre_views,
            post_views=post_views,
        )


@register_dataloader(name=dataset_name)
class MyDataLoader(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDataset,
            dataset_name=dataset_name,
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views,
        )
