import numpy as np
import torch
from scipy.sparse.linalg import eigsh
import pandas as pd
from torch.utils.data import DataLoader


def calculate_laplacian_matrix(adj_mat: np.ndarray, mat_type: str) -> np.ndarray:
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == "com_lap_mat":
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == "wid_rw_normd_lap_mat":
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which="LM", return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == "hat_rw_normd_lap_mat":
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(
            np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat
        )
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f"ERROR: {mat_type} is unknown.")


def maksed_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask_value: int = -1
) -> torch.Tensor:
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss

def get_norm_time(time: pd.Timestamp) -> float:
    time = pd.to_datetime(time, unit='s')
    hour = time.hour
    minute = time.minute

    if minute >= 30:
        hour = (hour + 1) % 24

    return hour / 24.0

def to_df(train_dataloader: DataLoader) -> pd.DataFrame:
    """transform train data to a dataframe"""
    df_list = []
    for sample in train_dataloader:
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