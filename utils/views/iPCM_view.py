import pandas as pd
from typing import Any

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("iPCM_preview")
def iPCM_preview(raw_df: pd.DataFrame, view_value: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from model.iPCM.iPCM_utils import build_time_period, build_region_id

    """
    A preprocessing view for iPCM that prepares the dataset for training.
    """
    logger.info("Applying iPCM_preview to dataset")

    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    view_value["num_users"] = num_users + 1
    view_value["num_pois"] = num_pois + 1

    time_period_list = build_time_period(raw_df)
    raw_df["time_period"] = time_period_list
    raw_df["time_period"] = pd.factorize(raw_df["time_period"])[0] + 1

    region_id_dict = build_region_id(raw_df, num_clusters=300)
    view_value["poi2region"] = region_id_dict
    raw_df["region_id"] = raw_df["POI_id"].map(region_id_dict)

    view_value["num_regions"] = len(set(region_id_dict.values())) + 1
    view_value["num_times"] = len(set(time_period_list)) + 1

    return raw_df, view_value
