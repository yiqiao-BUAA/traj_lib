import datetime
from datetime import timedelta
import random
import time
import pickle
from typing import Tuple, Any, List

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from utils.logger import get_logger
from model.CLSPRec.CLSPRec import CLSPRec
from utils.exargs import ConfigResolver
model_args = ConfigResolver("./model/CLSPRec/CLSPRec.yaml").parse()

log = get_logger(__name__)

pre_views = ["CLSPRec_preprocess"]  # Optional
post_views = []  # Optional
model = None


def generate_sample_to_device(sample: dict) -> list:
    """
    Convert sample data to device tensors.

    Args:
        sample (dict): Sample data containing user_id, POI_id, POI_catid, timestamps, and mask

    Returns:
        list: List of tensors for each feature moved to the appropriate device
    """
    device = model_args.device
    sample_to_device = []
    for i in range(len(sample["user_id"])):
        temp = []
        length = int(sample["mask"][i])
        temp.append(sample["POI_id"][i][:length])
        temp.append(sample["POI_catid"][i][:length])
        temp.append(torch.tensor([sample["user_id"][i] for j in range(length)]))
        hours = []
        for ts in sample["timestamps"][i][:length].tolist():
            # Convert milliseconds timestamp to seconds, then to datetime object
            dt = datetime.datetime.fromtimestamp(ts / 1000)
            hour = dt.hour
            hours.append(hour)
        temp.append(torch.tensor(hours))
        days = []
        for ts in sample["timestamps"][i][:length].tolist():
            # Convert milliseconds timestamp to seconds, then to datetime object
            dt = datetime.datetime.fromtimestamp(ts / 1000)
            day = dt.weekday() > 4
            days.append(day)
        temp.append(torch.tensor(days))
        sample_to_device.append(torch.stack(temp, dim=0).to(device))
    return sample_to_device


def generate_day_sample_to_device(day_trajectory: list) -> tuple:
    """
    Convert day trajectory data to device tensors.

    Args:
        day_trajectory (list): Day trajectory data

    Returns:
        tuple: (features tensor, day numbers tensor) on the appropriate device
    """
    device = model_args.device
    features = torch.tensor(day_trajectory[:5]).to(device)
    day_nums = torch.tensor(day_trajectory[5]).to(device)
    day_to_device = (features, day_nums)
    return day_to_device


def generate_negative_sample_list(
    dataloader: Any, idx: int
) -> list:
    """
    Generate negative samples for contrastive learning.
    
    Args:
        dataloader (DataLoader or list): Data loader or cached data list
        idx (int): Current sample index to avoid using as negative

    Returns:
        list: List of negative samples moved to the appropriate device
    """
    max_num = len(dataloader) - 1
    target_index = random.randint(0, max_num)
    while target_index == idx:
        target_index = random.randint(0, max_num)

    # Optimization: If dataloader is a list (cached data), access directly O(1)
    if isinstance(dataloader, list):
        item = dataloader[target_index]
        # Check if item is our cached tuple (sample, long_term_seq)
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], dict):
            sample = item[0]
        else:
            sample = item
        sample_to_device = generate_sample_to_device(sample)
    else:
        # Fallback for standard DataLoader iterator O(N)
        for index, sample in enumerate(dataloader):
            if index == target_index:
                sample_to_device = generate_sample_to_device(sample)
                break

    return sample_to_device


def build_user_history_map(raw_df: pd.DataFrame) -> dict:
    """
    Pre-process raw_df into a dictionary of user trajectories to speed up lookup.
    Logic: Group by User ID and Date (to simulate 'sequences').
    
    Returns:
        dict: {user_id: [{'ts': start_timestamp, 'data': dataframe_segment}, ... sorted by time]}
    """
    log.info("Building user history map from raw data...")
    # Ensure timestamps are datetime
    temp_df = raw_df.copy()
    temp_df['dt'] = pd.to_datetime(temp_df['timestamps'], unit='ms')
    temp_df['date'] = temp_df['dt'].dt.date
    
    # Group by User and Date (Traj definition)
    history_map = {}
    grouped = temp_df.groupby(['user_id', 'date'])
    
    for (uid, _), group in tqdm(grouped, desc="Grouping history"):
        if uid not in history_map:
            history_map[uid] = []
        
        # We store the dataframe segment and its start timestamp
        # Sorting within the day just in case
        sorted_group = group.sort_values('timestamps')
        start_ts = sorted_group['timestamps'].iloc[0]
        
        history_map[uid].append({
            'ts': start_ts,
            'data': sorted_group
        })
    
    # Sort trajectories for each user by time
    for uid in history_map:
        history_map[uid].sort(key=lambda x: x['ts'])
        
    return history_map


def generate_long_term_seq(sample: dict, history_map: dict, target_device=None) -> list:
    """
    Generate long-term sequences using specific logic:
    1. Short term seq length >= 5 (handled in dataset construction usually)
    2. Long term seqs are 7 days before the short term sequence
    3. Num long term seqs >= 2

    Args:
        sample (dict): Current batch sample
        history_map (dict): Pre-processed user history {user_id: [traj_objects]}
        target_device (torch.device, optional): Device to put tensors on.

    Returns:
        list: List of long-term sequences (concatenated tensors) for each user
    """


    # Constants matching the requirement
    PRE_SEQ_WINDOW_DAYS = 7
    MIN_LONG_TERM_COUNT = 2
    
    user_ids = sample["user_id"] # Tensor [Batch]
    # Use the first timestamp of the short-term sequence as the reference time "seq_time"
    # sample["timestamps"] is [Batch, SeqLen]
    seq_times = sample["timestamps"][:, 0] 
    
    long_term_seq_batch = []

    for i, user_id_tensor in enumerate(user_ids):
        user_id = user_id_tensor.item()
        current_seq_time = seq_times[i].item() # ms
        
        # Calculate window
        # start_time = seq_time - 7 days
        # end_time = seq_time
        current_dt = datetime.datetime.fromtimestamp(current_seq_time / 1000)
        start_dt = current_dt - timedelta(days=PRE_SEQ_WINDOW_DAYS)
        start_time_ms = start_dt.timestamp() * 1000
        
        user_trajs = history_map.get(user_id, [])
        
        valid_long_term_data = []
        
        # Find sequences in window [start_time, current_time]
        # Note: Original logic `start_time <= seq[0] <= end_time`
        # and we must exclude the current sequence itself (which starts at current_seq_time).
        # Assuming strict inequality < current_seq_time ensures we don't leak the target.
        for traj in user_trajs:
            traj_ts = traj['ts']
            if start_time_ms <= traj_ts < current_seq_time:
                valid_long_term_data.append(traj['data'])
        
        # Check count constraint
        if len(valid_long_term_data) >= MIN_LONG_TERM_COUNT:
            # Concatenate all valid long-term trajectories into one sequence
            merged_df = pd.concat(valid_long_term_data)
            
            # Convert to Tensor [5, Len] format
            seq = []
            # POI
            seq.append(torch.from_numpy(merged_df["POI_id"].values).to(target_device))
            # Cat
            seq.append(torch.from_numpy(merged_df["POI_catid"].values).to(target_device))
            # User
            seq.append(torch.from_numpy(merged_df["user_id"].values).to(target_device))
            # Hour
            seq.append(
                torch.from_numpy(
                    merged_df["dt"].dt.hour.values
                ).to(target_device)
            )
            # Day (Weekday > 4)
            seq.append(
                torch.from_numpy(
                    (merged_df["dt"].dt.dayofweek > 4).values
                ).to(target_device)
            )
            
            # Stack to [5, Len]
            long_term_seq_batch.append(torch.stack(seq, dim=0))
        else:
            long_term_seq_batch.append(None)
            
    return long_term_seq_batch


def train_model(
    train_set,
    test_set,
    model_args,
    vocab_size: dict,
    device: torch.device,
    raw_df: pd.DataFrame,
):
    """
    Train the CLSPRec model.
    """
    torch.cuda.empty_cache()

    # construct model
    global rec_model

    rec_model = rec_model.to(device)
    start_epoch = 0
    params = list(rec_model.parameters())
    optimizer = torch.optim.Adam(params, lr=model_args.lr)

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}

    # --- PREPROCESSING START ---
    # 1. Build History Map (Group by User/Day once)
    history_map = build_user_history_map(raw_df)

    # 2. Pre-calculate long-term sequences for training set
    log.info("Pre-calculating long-term sequences for training set...")
    cached_train_data = []
    
    # Iterate through the original DataLoader once
    # We use 'cpu' for storage to avoid OOM
    for sample in tqdm(train_set, desc="Pre-processing batches"):
        long_term_seq_cpu = generate_long_term_seq(sample, history_map, target_device='cpu')
        cached_train_data.append((sample, long_term_seq_cpu))
    
    log.info("Pre-calculation finished. Starting training...")
    # --- PREPROCESSING END ---

    for epoch in range(start_epoch, model_args.epoch):
        total_loss = 0.0
        
        # Shuffle cached batches
        random.shuffle(cached_train_data)
        
        pbar = tqdm(cached_train_data, desc=f"Epoch {epoch}", disable=True)
        
        for idx, (sample, long_term_seq_cpu) in enumerate(pbar):

            sample_to_device = generate_sample_to_device(sample)
            label = sample["y_POI_id"]["POI_id"].to(device)

            # Move cached long-term seqs to GPU
            long_term_seq = []
            for item in long_term_seq_cpu:
                if item is None:
                    long_term_seq.append(None)
                else:
                    long_term_seq.append(item.to(device))

            neg_sample_to_device_list = []
            if model_args["enable_ssl"]:
                neg_sample_to_device_list = generate_negative_sample_list(
                    cached_train_data, idx
                )

            loss, _ = rec_model(
                sample_to_device, long_term_seq, label, neg_sample_to_device_list
            )
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        # Note: For testing, we also need to generate long_term_seq using the map
        recall, ndcg, map, pred_raws, labels = test_model(test_set, rec_model, history_map)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss

        log.info(f"Epoch {epoch} done. Loss: {avg_loss:.4f}, Recall@10: {recall.get(10, 0):.4f}")

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if (
            len(past_10_loss) > 10
            and abs(total_loss - np.mean(past_10_loss)) < model_args.loss_delta
        ):
            break

def test_model(
    test_set, rec_model: CLSPRec, history_map: dict, ks: list = [1, 5, 10]
) -> tuple:
    """
    Test the CLSPRec model.
    Updated to take history_map instead of raw_df for consistency.
    """

    def calc_recall(labels: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels: torch.Tensor, preds: torch.Tensor, k: int) -> torch.Tensor:
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels, pred_raws = [], [], []
    for idx, sample in enumerate(test_set):
        sample_to_device = generate_sample_to_device(sample)
        device = model_args.device
        # Generate on-the-fly for test set (fast enough with history_map)
        long_term_seq = generate_long_term_seq(sample, history_map, target_device=device)
        
        label = sample["y_POI_id"]["POI_id"].to(device)
        neg_sample_to_device_list = []
        if model_args["enable_ssl"]:
            neg_sample_to_device_list = generate_negative_sample_list(test_set, idx)

        pred, label, pred_raw = rec_model.predict(
            sample_to_device, long_term_seq, label, neg_sample_to_device_list
        )
        preds.append(pred.detach())
        labels.append(label.detach())
        pred_raws.append(pred_raw.detach())
    preds = torch.cat(preds[:-1], dim=0)
    labels = torch.unsqueeze(torch.cat(labels[:-1], dim=0), -1)
    pred_raws = torch.cat(pred_raws[:-1], dim=0)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        

    return recalls, NDCGs, MAPs, pred_raws, labels


def train(train_dataloader: DataLoader, val_dataloader: DataLoader, view_value: dict[str, Any], **args):
    """
    Main training function.
    """
    global model
    global raw_df
    raw_df = view_value["raw_df"]
    now = datetime.datetime.now()

    # Get parameters
    device = model_args.device

    vocab_size = {
        "POI": torch.tensor(view_value["num_poi"]).to(device),
        "cat": torch.tensor(view_value["num_cat"]).to(device),
        "user": torch.tensor(view_value["num_user"]).to(device),
        "hour": torch.tensor(24).to(device),
        "day": torch.tensor(2).to(device),
    }
    global rec_model
    rec_model = CLSPRec(
        vocab_size=vocab_size,
        f_embed_size=model_args.embed_size,
        num_encoder_layers=model_args.tfp_layer_num,
        num_lstm_layers=model_args.lstm_layer_num,
        num_heads=model_args.head_num,
        forward_expansion=model_args.expansion,
        dropout_p=model_args.dropout,
    )

    
    train_model(
        train_dataloader,
        val_dataloader,
        model_args,
        vocab_size,
        device,
        raw_df,
    )



def inference(test_dataloader: DataLoader, view_value: dict, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform inference with the trained model.
    """
    global rec_model
    global raw_df
    
    # We need to build the history map for inference as well
    # Check if raw_df is available in view_value or global
    inference_raw_df = view_value.get("raw_df", raw_df)
    history_map = build_user_history_map(inference_raw_df)

    recalls, NDCGs, MAPs, pred_raws, labels = test_model(
        test_dataloader, rec_model, history_map
    )
    labels = labels.squeeze(1)
    pred_raws = pred_raws.cpu().numpy()
    labels = labels.cpu().numpy()

    return {'pred': pred_raws, 'gts': labels}