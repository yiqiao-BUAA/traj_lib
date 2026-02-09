import pandas as pd
from typing import Any

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("MTNet_preview")
def MTNet_preview(raw_df: pd.DataFrame, view_value: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from model.MTNet.MTNet_utils import build_region_id

    """
    A preprocessing view for MTNet that prepares the dataset for training.
    """
    logger.info("Applying MTNet_preview to dataset")

    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    num_poi_cats = raw_df["POI_catid"].nunique()
    view_value["num_users"] = num_users + 1
    view_value["num_pois"] = num_pois + 1
    view_value["num_poi_cats"] = num_poi_cats + 1

    region_list = build_region_id(raw_df, num_clusters=300)
    raw_df["region_id"] = region_list
    view_value["num_regions"] = len(set(region_list)) + 1

    return raw_df, view_value


@register_view("MTNet_post_view")
def MTNet_post_view(raw_df: list[dict[str, Any]], view_value: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    A preprocessing view for MTNet.
    """
    logger.info("Applying MTNet_post_view to dataset")

    for idx in range(len(raw_df)):
        raw_df[idx]["trajectory_id"] = idx + 1

    return raw_df, view_value
