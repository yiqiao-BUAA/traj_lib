import pandas as pd
from torch.utils.data import Dataset

import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("PRME_count")
def PRME_count(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view for PRME that prepares the dataset for training.
    """
    logger.info("Applying PRME_count to dataset")

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
