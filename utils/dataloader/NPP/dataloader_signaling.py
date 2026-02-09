from pathlib import Path

import pandas as pd

from utils.register import register_dataloader
from utils.logger import get_logger
from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
from utils.exargs import ConfigResolver
from utils.dataloader.NPP.decode import MemfdDataset

logger = get_logger(__name__)
base_name = "signaling"
root_path = Path(__file__).resolve().parent.parent.parent.parent
args = ConfigResolver(f"{root_path}/data/signaling/{base_name}.yaml").parse()


def decoder_step(name="signaling_original"):
    with MemfdDataset(
        enc_file=Path(f"{root_path}/data/signaling/signaling.enc"),
        key_yaml=Path(f"{root_path}/data/signaling/{base_name}.yaml"),
    ) as ds:
        with ds.open_fd(f"signaling/{name}.csv") as f:
            df = pd.read_csv(f)
    return df


def pre_process_func_original() -> pd.DataFrame:
    df = decoder_step("signaling_original")
    df["CID"] = pd.factorize(df["CID"])[0] + 1
    df["UID"] = pd.factorize(df["UID"])[0] + 1
    df.columns = [
        "user_id",
        "POI_id",
        "latitude",
        "longitude",
        "timestamps",
        "procedureEnd",
    ]
    df.drop("procedureEnd", axis="columns", inplace=True)
    df["POI_catid"] = 0
    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


def pre_process_func_simulated_2() -> pd.DataFrame:
    df = decoder_step("signaling_simulated_2")
    df["CID"] = pd.factorize(df["CID"])[0] + 1
    df["UID"] = pd.factorize(df["UID"])[0] + 1
    df.columns = [
        "user_id",
        "POI_id",
        "latitude",
        "longitude",
        "timestamps",
        "procedureEnd",
    ]
    df.drop("procedureEnd", axis="columns", inplace=True)
    df["POI_catid"] = 0
    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


def pre_process_func_simulated_59() -> pd.DataFrame:
    df = decoder_step("signaling_simulated_59")
    df["CID"] = pd.factorize(df["CID"])[0] + 1
    df["UID"] = pd.factorize(df["UID"])[0] + 1
    df.columns = [
        "user_id",
        "POI_id",
        "latitude",
        "longitude",
        "timestamps",
        "procedureEnd",
    ]
    df.drop("procedureEnd", axis="columns", inplace=True)
    df["POI_catid"] = 0
    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


def pre_process_func_simulated() -> pd.DataFrame:
    df = decoder_step("signaling_simulated")
    df["CID"] = pd.factorize(df["CID"])[0] + 1
    df["UID"] = pd.factorize(df["UID"])[0] + 1
    df.columns = [
        "user_id",
        "POI_id",
        "latitude",
        "longitude",
        "timestamps",
        "procedureEnd",
    ]
    df.drop("procedureEnd", axis="columns", inplace=True)
    df["POI_catid"] = 0
    df["user_id"] = df["user_id"].astype("int64")
    df["POI_id"] = df["POI_id"].astype("int64")
    df["POI_catid"] = df["POI_catid"].astype("int64")
    df = df.sort_values("timestamps")
    return df


class MyDataset(BaseDataset):
    def __init__(self, pre_views=None, post_views=None, type="original") -> None:
        if type == "original":
            super().__init__(
                preprocess_func=pre_process_func_original,
                sequence_length=args.get("sequence_length", 30),
                n_jobs=args.get("n_jobs", 1),
                pre_views=pre_views,
                post_views=post_views,
            )
        if type == "simulated_2":
            super().__init__(
                preprocess_func=pre_process_func_simulated_2,
                sequence_length=args.get("sequence_length", 30),
                n_jobs=args.get("n_jobs", 1),
                pre_views=pre_views,
                post_views=post_views,
            )
        if type == "simulated_59":
            super().__init__(
                preprocess_func=pre_process_func_simulated_59,
                sequence_length=args.get("sequence_length", 30),
                n_jobs=args.get("n_jobs", 1),
                pre_views=pre_views,
                post_views=post_views,
            )
        if type == "simulated":
            super().__init__(
                preprocess_func=pre_process_func_simulated,
                sequence_length=args.get("sequence_length", 30),
                n_jobs=args.get("n_jobs", 1),
                pre_views=pre_views,
                post_views=post_views,
            )


@register_dataloader(name="signaling_original")
class MyDataLoaderOriginal(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=MyDataset,
            dataset_name="signaling_original",
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views,
        )


@register_dataloader(name="signaling_simulated_2")
class MyDataLoader2(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=lambda pre_views, post_views: MyDataset(
                pre_views=pre_views, post_views=post_views, type="simulated_2"
            ),
            dataset_name="signaling_simulated_2",
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views,
        )


@register_dataloader(name="signaling_simulated_59")
class MyDataLoader59(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=lambda pre_views, post_views: MyDataset(
                pre_views=pre_views, post_views=post_views, type="simulated_59"
            ),
            dataset_name="signaling_simulated_59",
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views,
        )


@register_dataloader(name="signaling_simulated")
class MyDataLoaderSimulated(BaseDataLoader):
    def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
        super().__init__(
            MyDataset=lambda pre_views, post_views: MyDataset(
                pre_views=pre_views, post_views=post_views, type="simulated"
            ),
            dataset_name="signaling_simulated",
            logger=logger,
            args=args,
            model_args=model_args,
            pre_views=pre_views,
            post_views=post_views,
        )
