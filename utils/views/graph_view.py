import pandas as pd
from torch.utils.data import Dataset

import numpy as np

from utils.register import register_view
from utils.logger import get_logger

logger = get_logger(__name__)

def distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth specified in decimal degrees.
    """
    R = 6371.0  # Radius of the Earth in kilometers

    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

@register_view("adj_view")
def adj_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view that prepares the adjacency matrix for graph-based models.
    """
    logger.info("Applying adj_view to dataset")
    common_count_value = ['n_user', 'n_item']
    for key in common_count_value:
        if key not in view_value:
            logger.error(f"View value must contain '{key}' calculated from 'common_count' view.")
            raise KeyError(f"View value must contain '{key}' calculated from 'common_count' view.")
    
    n_item = view_value["n_item"]

    # Initialize adjacency matrix
    # w_ij = \exp(\frac{-dist_ij^2}{sigma^2}) if i != j else 0
    adj_matrix = np.zeros((n_item, n_item), dtype=np.float32)
    #cul late distance between POIs by latitude and longitude
    lats, lons = [], []
    poi_coords = {}
    for i, row in raw_df.iterrows():
        poi_id = int(row["POI_id"])
        poi_coords[poi_id] = (row["latitude"], row["longitude"])

    lats = np.array([poi_coords.get(i, (0.0, 0.0))[0] for i in range(n_item)])
    lons = np.array([poi_coords.get(i, (0.0, 0.0))[1] for i in range(n_item)]).T
    
    sigma2 = 10.0          # hyperparameter in Paper
    epsilon = 0.5

    dist = distance(lats[:, None], lons[:, None], lats[None, :], lons[None, :])  # d_ij
    adj_matrix = np.exp(-(dist ** 2) / sigma2)

    adj_matrix[adj_matrix < epsilon] = 0.0
    np.fill_diagonal(adj_matrix, 0.0)

    view_value["adj"] = adj_matrix

    return raw_df, view_value

@register_view("subgraph_view")
def subgraph_view(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A preprocessing view that prepares subgraphs for each user in the dataset.
    """
    logger.info("Applying subgraph_view to dataset")
    common_count_value = ['n_user', 'n_item']
    for key in common_count_value:
        if key not in view_value:
            logger.error(f"View value must contain '{key}' calculated from 'common_count' view.")
            raise KeyError(f"View value must contain '{key}' calculated from 'common_count' view.")
    
    n_item = view_value["n_item"]

    # Create a dictionary to hold subgraphs for each poi by distance

    subgraph_point = 100 # k-nearest neighbors

    subgraphs = {i: set() for i in range(n_item)}
    lats, lons = [], []
    poi_coords = {}
    for i, row in raw_df.iterrows():
        poi_id = int(row["POI_id"])
        poi_coords[poi_id] = (row["latitude"], row["longitude"])
    lats = np.array([poi_coords.get(i, (0.0, 0.0))[0] for i in range(n_item)])
    lons = np.array([poi_coords.get(i, (0.0, 0.0))[1] for i in range(n_item)])

    dist = distance(lats[:, None], lons[:, None], lats[None, :], lons[None, :])  # d_ij
    for i in range(n_item):
        nearest_indices = np.argsort(dist[i])[1:subgraph_point + 1]  # +1 to exclude self
        subgraphs[i] = set(nearest_indices.tolist())

    view_value["subgraphs"] = subgraphs
    return raw_df, view_value