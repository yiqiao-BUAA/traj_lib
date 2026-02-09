import pandas as pd
import numpy as np
from typing import Any, Tuple

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)


@register_view("ROTAN_post_view")
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


@register_view("rotan_preview")
def rotan_preview(raw_df: pd.DataFrame, view_value: dict[str, Any]) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """
    Time-aware preprocessing view for ROTAN with TimeModel.

    Produces the fields required by TimeModel:
      - hod   : hour-of-day in [0,24) (float)
      - dow   : day-of-week in [0,7) (float, Mon=0)
      - doy   : day-of-year in [1,366] (float)
      - delta_t: inter-arrival gap in HOURS, per user, >=0 (float)
      - t_norm: absolute time normalized to [0,1] using a global (min,max)

    Also sets:
      - view_value["num_users"], view_value["num_pois"] ( +1 for PAD )
      - view_value["time_min"], view_value["time_max"] for consistent normalization across splits

    Notes:
      * If view_value already contains time_min/time_max, they will be used; otherwise we compute them on-the-fly.
      * Keeps your existing quad_key construction if available.
    """
    logger.info("Applying rotan_preview (time-aware) to dataset")

    # ---- ID vocab sizes (with +1 for PAD) ----
    num_users = raw_df["user_id"].nunique()
    num_pois = raw_df["POI_id"].nunique()
    view_value["num_users"] = int(num_users) + 1
    view_value["num_pois"] = int(num_pois) + 1

    # ---- Ensure timestamps are datetime ----
    ts = pd.to_datetime(raw_df["timestamps"], errors="coerce", utc=True)
    if ts.isna().any():
        n_bad = int(ts.isna().sum())
        logger.warning(f"rotan_preview: {n_bad} rows have invalid timestamps; dropping them.")
        valid_mask = ~ts.isna()
        raw_df = raw_df.loc[valid_mask].copy()
        ts = ts.loc[valid_mask]

    # Use timezone-naive UTC for deterministic extraction
    ts_naive = ts.dt.tz_convert(None)
    raw_df["timestamps"] = ts_naive

    # ---- Per-row time features ----
    hr = ts_naive.dt.hour.astype(np.float32)
    mn = ts_naive.dt.minute.astype(np.float32)
    raw_df["hod"] = hr + mn / 60.0
    raw_df["dow"] = ts_naive.dt.dayofweek.astype(np.float32)  # Mon=0
    raw_df["doy"] = ts_naive.dt.dayofyear.astype(np.float32)

    # ---- delta_t in HOURS (per user, sorted by time) ----
    # sort for stable diff
    raw_df = raw_df.sort_values(["user_id", "timestamps"]).reset_index(drop=True)
    # seconds since epoch for diff
    sec = (raw_df["timestamps"].astype("int64") // 10**9).astype("int64")
    # groupby diff
    delta_sec = sec.groupby(raw_df["user_id"]).diff().fillna(0)
    delta_sec = delta_sec.clip(lower=0)
    raw_df["delta_t"] = (delta_sec / 3600.0).astype(np.float32)

    # ---- global (min,max) for t_norm ----
    # Prefer provided bounds (for validation/test consistency)
    t_min = view_value.get("time_min", None)
    t_max = view_value.get("time_max", None)
    if t_min is None or t_max is None:
        t_min = int(sec.min())
        t_max = int(sec.max())
        # avoid degenerate case
        if t_max <= t_min:
            t_max = t_min + 1
        view_value["time_min"] = t_min
        view_value["time_max"] = t_max
    denom = float(max(1, t_max - t_min))
    raw_df["t_norm"] = ((sec - t_min) / denom).astype(np.float32).clip(0.0, 1.0)

    # ---- Optional: quad_key (reuse your existing util if present) ----
    try:
        from model.ROTAN.ROTAN_utils import get_all_permutations_dict, get_quad_keys
        permutations_dict = get_all_permutations_dict(6)
        lats = raw_df["latitude"].to_numpy()
        lons = raw_df["longitude"].to_numpy()
        raw_df["quad_key"] = [
            get_quad_keys(lat, lon, permutations_dict) for lat, lon in zip(lats, lons)
        ]
    except Exception as e:
        logger.warning(f"rotan_preview: quad_key generation skipped due to error: {e}")

    # ---- Keep an integer representation if downstream needs it ----
    raw_df["timestamps"] = (raw_df["timestamps"].astype("int64"))  # nanoseconds since epoch

    # Cast to minimal dtypes where appropriate
    for col in ["hod", "dow", "doy", "delta_t", "t_norm"]:
        raw_df[col] = raw_df[col].astype(np.float32)

    return raw_df, view_value
