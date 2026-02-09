# from pathlib import Path
# from typing import Callable

# import pandas as pd

# from utils.register import register_dataloader
# from utils.logger import get_logger
# from utils.dataloader.NPP.dataloader_base import BaseDataset, BaseDataLoader
# from utils.exargs import ConfigResolver

# logger = get_logger(__name__)
# dataset_name = "yjmob"
# root_path = Path(__file__).resolve().parent.parent.parent.parent
# args = ConfigResolver(f"{root_path}/data/{dataset_name}/{dataset_name}.yaml").parse()


# def pre_process_func1() -> pd.DataFrame:
#     df = pd.read_csv(f"{root_path}/data/{dataset_name}/yjmob100k-dataset1.csv")
#     df = df[['uid', 'd', 't', 'x', 'y']]
#     df = df.rename(columns={"x": "longitude", "y": "latitude", "uid": "user_id"})
#     df['d'] = df['d'].astype('int64')
#     df['t'] = df['t'].astype('int64')
#     df['latitude'] = df['latitude'].astype('float64')
#     df['longitude'] = df['longitude'].astype('float64')
#     df['timestamps'] = df['d'] * 48 + df['t']
#     df['trajectory_id'] = 0
#     x_max = df['longitude'].max()
#     df['POI_id'] = df['latitude'] * x_max + df['longitude']
#     df['POI_id'] = pd.factorize(df['POI_id'])[0]
#     df["POI_catid"] = 0
    
#     df = df.sort_values("timestamps")
#     return df

# def pre_process_func2() -> pd.DataFrame:
#     df = pd.read_csv(f"{root_path}/data/{dataset_name}/yjmob100k-dataset2.csv")
#     df = df[['uid', 'd', 't', 'x', 'y']]
#     df = df.rename(columns={"x": "longitude", "y": "latitude", "uid": "user_id"})
#     df['d'] = df['d'].astype('int64')
#     df['t'] = df['t'].astype('int64')
#     df['latitude'] = df['latitude'].astype('float64')
#     df['longitude'] = df['longitude'].astype('float64')
#     df['timestamps'] = df['d'] * 48 + df['t']
#     df['trajectory_id'] = 0
#     x_max = df['longitude'].max()
#     df['POI_id'] = df['latitude'] * x_max + df['longitude']
#     df['POI_id'] = pd.factorize(df['POI_id'])[0]
#     df["POI_catid"] = 0
#     df = df.sort_values("timestamps")
#     return df

# class MyDataset(BaseDataset):
#     def __init__(
#         self,
#         pre_views: Callable=None,
#         post_views: Callable=None,
#         type: str=None
#     ) -> None:
#         if type == "1":
#             super().__init__(
#                 preprocess_func=pre_process_func1,
#                 sequence_length=args.get("sequence_length", 30),
#                 n_jobs=args.get("n_jobs", 1),
#                 pre_views=pre_views,
#                 post_views=post_views,
#             )
#         elif type == "2":
#             super().__init__(
#                 preprocess_func=pre_process_func2,
#                 sequence_length=args.get("sequence_length", 30),
#                 n_jobs=args.get("n_jobs", 1),
#                 pre_views=pre_views,
#                 post_views=post_views,
#             )
#         else:
#             raise ValueError(f"Unknown type: {type}")

# @register_dataloader(name=f'{dataset_name}_1')
# class MyDataLoader1(BaseDataLoader):
#     def __init__(
#         self,
#         model_args: dict=None,
#         pre_views: Callable=None,
#         post_views: Callable=None
#     ) -> None:
#         super().__init__(
#             MyDataset=lambda pre_views, post_views: MyDataset(
#                 pre_views=pre_views, post_views=post_views, type="1"
#             ),
#             dataset_name=dataset_name,
#             logger=logger,
#             args=args,
#             model_args=model_args,
#             pre_views=pre_views,
#             post_views=post_views,
#         )

# @register_dataloader(name=f'{dataset_name}_2')
# class MyDataLoader2(BaseDataLoader):
#     def __init__(
#         self,
#         model_args: dict=None,
#         pre_views: Callable=None,
#         post_views: Callable=None
#     ) -> None:
#         super().__init__(
#             MyDataset=lambda pre_views, post_views: MyDataset(
#                 pre_views=pre_views, post_views=post_views, type="2"
#             ),
#             dataset_name=dataset_name,
#             logger=logger,
#             args=args,
#             model_args=model_args,
#             pre_views=pre_views,
#             post_views=post_views,
#         )


# if __name__ == "__main__":
#     dict_args = dict()
#     d = MyDataLoader1(model_args=dict_args)
#     train_dataloader = d.train_dataloader
#     for i, batch in enumerate(train_dataloader):
#         print(batch)
#         break
#     dict_args = dict()
#     d = MyDataLoader2(model_args=dict_args)
#     train_dataloader = d.train_dataloader
#     for i, batch in enumerate(train_dataloader):
#         print(batch)
#         exit(0)
