import pandas as pd
import numpy as np
from typing import Any

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("ROTAN_preview")
def ROTAN_preview(raw_df: pd.DataFrame, view_value: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from model.ROTAN.ROTAN_utils import (
        get_norm_time96,
        get_day_norm7,
        get_time_slot_id,
        get_all_permutations_dict,
        get_quad_keys,
    )

    """
    A preprocessing view for ROTAN that prepares the dataset for training.
    """
    logger.info("Applying ROTAN_preview to dataset")

    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    view_value["num_users"] = num_users + 1
    view_value["num_pois"] = num_pois + 1

    raw_df["norm_time"] = raw_df["timestamps"].apply(get_norm_time96)
    raw_df["day_time"] = raw_df["timestamps"].apply(get_day_norm7)
    raw_df["time_id"] = raw_df["timestamps"].apply(get_time_slot_id)

    permutations_dict = get_all_permutations_dict(6)
    raw_df["quad_key"] = raw_df.apply(
        lambda row: get_quad_keys(row["latitude"], row["longitude"], permutations_dict),
        axis=1,
    )

    return raw_df, view_value


@register_view("ROTAN_postview")
def ROTAN_post_view(raw_df: list[dict[str, Any]], view_value: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    A preprocessing view for ROTAN.
    """
    logger.info("Applying ROTAN_post_view to dataset")

    for seq_data in raw_df:
        quad_key = seq_data["quad_key"]
        new_quad_key = []
        for quad in quad_key:
            if isinstance(quad, list):
                new_quad_key.append(quad)
            else:
                new_quad_key.append([0] * len(quad_key[0]))
        new_quad_key_array: np.ndarray = np.array(new_quad_key)
        seq_data["quad_key"] = new_quad_key_array

    return raw_df, view_value
