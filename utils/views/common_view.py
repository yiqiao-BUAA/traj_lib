import pandas as pd
from torch.utils.data import Dataset

import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("common_count")
def common_count(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view that prepares the dataset for training.
    """
    logger.info("Applying common_count to dataset")

    # count the number of unique users and items
    n_user = raw_df["user_id"].nunique()
    n_item = raw_df["POI_id"].nunique()

    view_value["n_user"] = n_user + 1
    view_value["n_item"] = n_item + 1

    poi_coords = np.zeros((n_item + 2, 2), dtype=np.float32)

    for i, row in raw_df.iterrows():
        poi_id = int(row["POI_id"])
        poi_coords[poi_id] = np.array(
            [row["longitude"], row["latitude"]], dtype=np.float32
        )

    view_value["poi_coords"] = poi_coords

    return raw_df, view_value

@register_view("seq_len")
def seq_len_view(raw_df: list[dict], view_value: dict = None) -> tuple[list[dict], dict]:
    """
    A preprocessing view that calculates the maximum sequence length in the dataset.
    """
    logger.info("Applying seq_len_view to dataset")

    # Calculate the maximum sequence length
    seq_lengths = len(raw_df[0]["POI_id"])

    view_value["seq_len"] = seq_lengths

    logger.info(f"Calculated maximum sequence length: {seq_lengths}")

    return raw_df, view_value