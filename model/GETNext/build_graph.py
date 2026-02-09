import networkx as nx
import pandas as pd
from typing import Any
from tqdm import tqdm


def build_global_POI_checkin_graph(df: pd.DataFrame, exclude_user=None) -> Any:
    G = nx.DiGraph()
    users = list(set(df["user_id"].to_list()))
    if exclude_user in users:
        users.remove(exclude_user)
    loop = tqdm(users)
    for user_id in loop:
        user_df = df[df["user_id"] == user_id]

        # Add node (POI)
        for i, row in user_df.iterrows():
            node = row["POI_id"]
            if node not in G.nodes():
                G.add_node(
                    row["POI_id"],
                    checkin_cnt=1,
                    poi_catid=row["POI_catid"],
                    poi_catid_code=0,
                    poi_catname="",
                    latitude=row["latitude"],
                    longitude=row["longitude"],
                )
            else:
                G.nodes[node]["checkin_cnt"] += 1

        # Add edges (Check-in seq)
        previous_poi_id = 0
        previous_traj_id = 0
        for i, row in user_df.iterrows():
            poi_id = row["POI_id"]
            traj_id = row["user_id"]
            # No edge for the begin of the seq or different traj
            if (previous_poi_id == 0) or (previous_traj_id != traj_id):
                previous_poi_id = poi_id
                previous_traj_id = traj_id
                continue

            # Add edges
            if G.has_edge(previous_poi_id, poi_id):
                G.edges[previous_poi_id, poi_id]["weight"] += 1
            else:  # Add new edge
                G.add_edge(previous_poi_id, poi_id, weight=1)
            previous_traj_id = traj_id
            previous_poi_id = poi_id

    return G
