# from pathlib import Path

# import pandas as pd

# from utils.register import register_dataloader
# from utils.logger import get_logger
# from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
# from utils.exargs import ConfigResolver

# logger = get_logger(__name__)
# dataset_name = "xian"
# root_path = Path(__file__).resolve().parent.parent.parent.parent
# args = ConfigResolver(f"{root_path}/data/{dataset_name}/{dataset_name}.yaml").parse()


# def pre_process_func() -> pd.DataFrame:
#     df = pd.read_csv(f"{root_path}/data/{dataset_name}/{dataset_name}.csv")
#     df = df[["GPS时间", "POI_id", "车辆ID", "订单ID", "latitude", "longitude"]]
#     df["POI_id"] = pd.factorize(df["POI_id"])[0] + 1
#     df["车辆ID"] = pd.factorize(df["车辆ID"])[0] + 1
#     df["订单ID"] = pd.factorize(df["订单ID"])[0] + 1
#     df["GPS时间"] = (
#         pd.to_datetime(df["GPS时间"], format="%Y-%m-%d %H:%M:%S", utc=True).astype(
#             "int64"
#         )
#         // 10**9
#     )
#     df.columns = [
#         "timestamps",
#         "POI_id",
#         "user_id",
#         "trajectory_id",
#         "latitude",
#         "longitude",
#     ]
#     df["POI_catid"] = 0
#     df["user_id"] = df["user_id"].astype("int64")
#     df["POI_id"] = df["POI_id"].astype("int64")
#     df["POI_catid"] = df["POI_catid"].astype("int64")
#     df = df.sort_values("timestamps")
#     return df


# class MyDataset(BaseDataset):
#     def __init__(self, pre_views=None, post_views=None) -> None:
#         super().__init__(
#             preprocess_func=pre_process_func,
#             sequence_length=args.get("sequence_length", 30),
#             n_jobs=args.get("n_jobs", 1),
#             pre_views=pre_views,
#             post_views=post_views,
#         )


# @register_dataloader(name=dataset_name)
# class MyDataLoader(BaseDataLoader):
#     def __init__(self, model_args=None, pre_views=None, post_views=None) -> None:
#         super().__init__(
#             MyDataset=MyDataset,
#             dataset_name=dataset_name,
#             logger=logger,
#             args=args,
#             model_args=model_args,
#             pre_views=pre_views,
#             post_views=post_views,
#         )
