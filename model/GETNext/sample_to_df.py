import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm


def to_df(train_dataloader: DataLoader) -> pd.DataFrame:
    """transform train data to a dataframe"""
    df_list = []
    for sample in tqdm(train_dataloader):
        user_id = sample["user_id"]
        POI_id = sample["POI_id"]
        longitude = sample["longitude"]
        latitude = sample["latitude"]
        POI_catid = sample["POI_catid"]
        timestamps = sample["timestamps"]
        y = sample["y_POI_id"]
        for i in range(user_id.shape[0]):
            for j in range(POI_id.shape[1]):
                if int(POI_id[i][j]) != 0:
                    temp_list = [
                        int(user_id[i]),
                        int(POI_id[i][j]),
                        int(timestamps[i][j]),
                        float(latitude[i][j]),
                        float(longitude[i][j]),
                        int(POI_catid[i][j]),
                    ]
                    df_list.append(temp_list)
                else:
                    break
            temp_list = [
                int(y["user_id"][i]),
                int(y["POI_id"][i]),
                int(y["timestamps"][i]),
                float(y["latitude"][i]),
                float(y["longitude"][i]),
                int(y["POI_catid"][i]),
            ]
            df_list.append(temp_list)
    df = pd.DataFrame(
        data=df_list,
        columns=[
            "user_id",
            "POI_id",
            "timestamps",
            "latitude",
            "longitude",
            "POI_catid",
        ],
    )
    return df
