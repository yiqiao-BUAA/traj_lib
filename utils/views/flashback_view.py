from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

import pandas as pd
import numpy as np


def dis_cul(lonx, latx, lony, laty):
    """
    Calculate the distance between two points given their longitude and latitude.
    """
    from math import radians, cos, sin, asin, sqrt

    # Convert degrees to radians
    lonx, latx, lony, laty = map(radians, [lonx, latx, lony, laty])
    # Haversine formula
    dlon = lony - lonx
    dlat = laty - latx
    a = sin(dlat / 2) ** 2 + cos(latx) * cos(laty) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


@register_view("flashback_poi_graph")
def flashback_poi_graph(
    raw_df: pd.DataFrame, view_value: dict = None
) -> tuple[pd.DataFrame, dict]:
    """
    A preprocessing view for Flashback that prepares the POI graph.
    """
    logger.info("Applying flashback_poi_graph to dataset")

    # Create a transition graph from the raw DataFrame
    POI_dict, cnt = {}, 0
    for _, row in raw_df.iterrows():
        POI_id = row["POI_id"]
        if POI_id not in POI_dict:
            POI_dict[POI_id] = [cnt, row["longitude"], row["latitude"]]
            cnt += 1
        row["POI_id"] = POI_dict[POI_id]

    graph = np.zeros((cnt, cnt), dtype=np.float32)
    for i in range(cnt):
        for j in range(cnt):
            if i != j:
                dist = dis_cul(
                    POI_dict[i][1], POI_dict[i][2], POI_dict[j][1], POI_dict[j][2]
                )
                graph[i][j] = dist
    view_value["transition_graph"] = graph
