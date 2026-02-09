from utils.register import register_view
from utils.logger import get_logger
logger = get_logger(__name__)
import pandas as pd
from torch.utils.data import Dataset

@register_view("analyze_POI")
def analyze_POI(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    poi_counts = raw_df['POI_id'].value_counts().to_dict()
    view_value['poi_counts'] = poi_counts
    # bucket POIs by frequency top-1%， 1-10%， 10%-50%， >50%
    total_pois = len(poi_counts)
    buckets = {'0-1%': [], '1%-10%': [], '10%-50%': [], '>50%': []}
    for poi, count in poi_counts.items():
        rank = sorted(poi_counts.values(), reverse=True).index(count) + 1
        percentile = rank / total_pois
        if percentile <= 0.01:
            buckets['0-1%'].append(poi)
        elif percentile <= 0.1:
            buckets['1%-10%'].append(poi)
        elif percentile <= 0.5:
            buckets['10%-50%'].append(poi)
        else:
            buckets['>50%'].append(poi)
    view_value['poi_buckets'] = buckets
    logger.info(f"POI Buckets: { {k: len(v) for k, v in buckets.items()} }")
    return raw_df, view_value