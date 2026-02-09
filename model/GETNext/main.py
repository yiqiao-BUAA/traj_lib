from datetime import datetime
from argparse import Namespace
from typing import Tuple, List, Any, Optional, Dict, Callable, Iterable

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logger import get_logger
from model.GETNext.build_graph import build_global_POI_checkin_graph
from utils.exargs import ConfigResolver
from utils.GPU_find import find_gpu
from model.GETNext.model import GETNextModel
from model.GETNext.utils import (
    calculate_laplacian_matrix,
    maksed_mse_loss,
    to_df,
)


log = get_logger(__name__)

pre_views = ["GETNext_view"]

# global variables
model : GETNextModel | None = None
X, A = None, None

def build_graph(train_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build a graph from training data and extract adjacency matrix and node features.

    Args:
        train_df: DataFrame containing training data with POI check-in information

    Returns:
        Tuple containing:
        - A: Adjacency matrix as numpy array
        - X: Node features DataFrame with columns for POI attributes
    """
    # Build global POI check-in graph
    G = build_global_POI_checkin_graph(train_df)

    # Get node list and adjacency matrix
    nodelist = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodelist)

    # Extract node attributes efficiently
    nodes_data = list(G.nodes(data=True))

    # Use list comprehension for faster processing
    temp_list = [
        [
            node,
            data["checkin_cnt"],
            data["poi_catid"],
            data["poi_catid_code"],
            data["poi_catname"],
            data["latitude"],
            data["longitude"],
        ]
        for node, data in nodes_data
    ]

    # Create DataFrame with specified columns
    X = pd.DataFrame(
        temp_list,
        columns=[
            "node_name/poi_id",
            "checkin_cnt",
            "poi_catid",
            "poi_catid_code",
            "poi_catname",
            "latitude",
            "longitude",
        ],
    )

    return A, X


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    view_value: dict,
    eval_funcs: dict[str, Callable],
    **kwargs,
) -> Iterable[dict[str, Any]]:
    args = ConfigResolver(f"model/GETNext/GETNext.yaml").parse()
    args.update({'device': find_gpu()})

    train_loader = train_dataloader
    val_loader = val_dataloader
    train_df = to_df(train_loader)
    raw_A, X_df = build_graph(train_df=train_df)
    global X, A
    num_users, num_pois, num_cats = view_value["num_users"], view_value["num_pois"], view_value["num_cats"]
    log.info(f"Number of users: {num_users}, Number of POIs: {num_pois}, Number of POI categories: {num_cats}")
    feature1 = "checkin_cnt"
    feature2 = "poi_catid"
    feature3 = "latitude"
    feature4 = "longitude"
    raw_X = X_df[[feature1, feature2, feature3, feature4]].to_numpy()

    log.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {feature1}, {feature2}, {feature3}, {feature4}."
    )
    log.info(
        f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency)."
    )
    
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(
        list(map(lambda x: [x], cat_list))
    ).toarray()
    one_hot_emb = np.zeros((one_hot_rlt.shape[0], num_cats) , dtype=np.float32)
    one_hot_emb[:, :one_hot_rlt.shape[1]] = one_hot_rlt
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:raw_X.shape[0], 0] = raw_X[:, 0]
    X[:raw_X.shape[0], 1 : num_cats + 1] = one_hot_emb
    X[:raw_X.shape[0], num_cats + 1 :] = raw_X[:, 2:]
    log.info(f"After one hot encoding poi cat, X.shape: {X.shape}")

    # Normalization
    A = calculate_laplacian_matrix(raw_A, mat_type="hat_rw_normd_lap_mat")
    full_A = np.zeros((num_pois, num_pois), dtype=np.float32)
    full_A[:A.shape[0], :A.shape[1]] = A
    A = full_A

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)
    args.gcn_nfeat = X.shape[1]
    
    global model

    args.seq_input_embed = (
            args.poi_embed_dim
            + args.user_embed_dim
            + args.time_embed_dim
            + args.cat_embed_dim
        )
    model = GETNextModel(node_attn_in_features=X.shape[1], node_attn_nhid=args.node_attn_nhid, 
                         gcn_ninput=args.gcn_nfeat, gcn_nhid=args.gcn_nhid, gcn_dropout=args.gcn_dropout,
                         seq_input_embed=args.seq_input_embed, transformer_dropout=args.transformer_dropout,
                         transformer_nhead=args.transformer_nhead,transformer_nhid=args.transformer_nhid, transformer_nlayers=args.transformer_nlayers, 
                         poi_embed_dim=args.poi_embed_dim, user_embed_dim=args.user_embed_dim, 
                         time_embed_dim=args.time_embed_dim, cat_embed_dim=args.cat_embed_dim, 
                         num_cats=num_cats, num_users=num_users, num_pois=num_pois, device=args.device)

    # Define overall loss and optimizer
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=args.lr_scheduler_factor)
    # %% ====================== Train ======================
    model = model.to(device=args.device)
    for epoch in range(args.epochs):
        loss_list = []
        model.train()

        for b_idx, batch in enumerate(tqdm(train_loader)):

            y_pred_poi, y_pred_time, y_pred_cat = model(batch, X, A)
            
            B, L = batch["POI_id"].size()
            y_poi_seq = torch.full((B, L), 0, dtype=torch.long, device=args.device)
            y_time_seq = torch.full((B, L), 0, dtype=torch.float, device=args.device)
            y_cat_seq = torch.full((B, L), 0, dtype=torch.long, device=args.device)
            poi_id = batch["POI_id"]
            timestamps = batch["norm_time"]
            cat_id = batch["POI_catid"]
            y_poi_id, y_norm_time, y_cat = batch['y_POI_id']['POI_id'], batch['y_POI_id']['norm_time'], batch['y_POI_id']['POI_catid']
            for i in range(B):
                end = batch["mask"][i].item()
                y_poi_seq[i, :end] = torch.cat(
                    (poi_id[i, 1:end], y_poi_id[i].unsqueeze(dim=-1)), dim=-1
                )
                y_time_seq[i, :end] = torch.cat(
                    (timestamps[i, 1:end], y_norm_time[i].unsqueeze(dim=-1)), dim=-1
                )
                y_cat_seq[i, :end] = torch.cat(
                    (cat_id[i, 1:end], y_cat[i].unsqueeze(dim=-1)), dim=-1
                )

            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi_seq)
            loss_time = criterion_time(y_pred_time.squeeze(), y_time_seq)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat_seq)

            # Final loss
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_list.append(loss.item())
        # train end --------------------------------------------------------------------------------------------------------
        res_dict  = inference(val_loader, view_value, **kwargs)
        pred, gt, loss = res_dict['pred'], res_dict['gts'], res_dict['loss']
        scores = {}
        for name, func in eval_funcs.items():
            score = func(pred, gt)
            scores[name] = score
        log.info(f"Epoch {epoch+1}/{args.epochs}, loss: {sum(loss_list)/len(loss_list):.4f}")
        lr_scheduler.step(loss)
        yield [scores, {'loss': sum(loss_list)/len(loss_list)}]
        # valid end --------------------------------------------------------------------------------------------------------


def inference(
    test_dataloader: DataLoader, view_value: dict, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    final_pred = []
    final_y = []
    loss_list = []
    model_args = ConfigResolver(f"model/GETNext/GETNext.yaml").parse()
    args = Namespace(**model_args)
    global model
    global X, A
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    model.eval()

    with torch.no_grad():
        for vb_idx, batch in enumerate(test_dataloader):
            
            y_pred_poi, _, _ = model(batch, X, A)
            y_pred_list = []
            for idx, len in enumerate(batch['mask']):
                y_pred_list.append(y_pred_poi[idx, int(len)-1, :].cpu().detach().numpy())
            y_pred_poi = torch.Tensor(np.array(y_pred_list))
            y_poi = batch['y_POI_id']['POI_id']
            
            loss_poi = criterion_poi(y_pred_poi, y_poi)
            loss_list.append(loss_poi.item())
            final_pred.append(y_pred_poi)
            final_y.append(y_poi)

    final_y_pred_poi = torch.cat(final_pred, dim=0).detach().cpu().numpy()
    final_y_poi = torch.cat(final_y, dim=0).detach().cpu().numpy()

    return {'pred': final_y_pred_poi, 'gts': final_y_poi, 'loss': np.mean(loss_list)} # [batch, n_items], [batch], float
