import pandas as pd
import numpy as np
from typing import Any

from utils.register import register_view
from utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)


@register_view("TPG_preview")
def TPG_preview(raw_df: pd.DataFrame, view_value: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    from model.TPG.TPG_utils import build_region_id, QuadkeyField

    """
    A preprocessing view for TPG that prepares the dataset for training.
    """
    logger.info("Applying TPG_preview to dataset")

    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    num_poi_cats = raw_df["POI_catid"].nunique()
    view_value["num_users"] = num_users + 1
    view_value["num_pois"] = num_pois + 1
    view_value["num_poi_cats"] = num_poi_cats + 1

    raw_df["time_id"] = raw_df["timestamps"].apply(
        lambda x: datetime.fromtimestamp(x).weekday() * 24
        + datetime.fromtimestamp(x).hour
        + 1
    )
    view_value["region_id_map"], QUADKEY = build_region_id(
        raw_df["POI_id"].tolist(),
        raw_df["latitude"].tolist(),
        raw_df["longitude"].tolist(),
    )
    view_value["num_time"] = raw_df["time_id"].nunique() + 1

    global_quadkeys = QuadkeyField()
    global_quadkeys.build_vocab(QUADKEY)
    view_value["QUADKEY"] = global_quadkeys
    view_value["nquadkey"] = len(global_quadkeys.vocab)

    return raw_df, view_value


@register_view("TPG_post_view")
def TPG_post_view(raw_df: list[dict[str, Any]], view_value: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from tqdm import tqdm

    """
    A post view for TPG that prepares the dataset for training.
    """
    logger.info("Applying TPG_post_view to dataset")

    global_quadkeys = view_value["QUADKEY"]

    length = 9

    for seq_data in tqdm(raw_df):
        region: list = []
        for _ in range(length):
            region.append([])
        region_seq = [
            view_value["region_id_map"].get(poi_id, None)
            for poi_id in seq_data["POI_id"]
        ]
        for idx in range(length):
            r = [r_[idx] for r_ in region_seq]
            r_tuple = tuple(r)
            r_global = global_quadkeys.numericalize(list(r_tuple))  # (L, LEN_QUADKEY)
            region[idx].append(r_global)
        # length, seq_len, LEN_QUADKEY -> seq_len, length, LEN_QUADKEY
        seq_data["region_id"] = np.concatenate(region, axis=0).transpose(1, 0, 2)

    return raw_df, view_value
