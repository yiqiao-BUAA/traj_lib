from typing import Tuple, Optional, Callable, Any, Sequence
from collections.abc import Iterable

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from utils.GPU_find import find_gpu
from .iPCM import iPCM
from .iPCM_utils import construct_graph

args = ConfigResolver("./model/iPCM/iPCM.yaml").parse()
log = get_logger(__name__)

pre_views = ["iPCM_preview"]
post_views: list[str] = []

model: Optional[iPCM] = None
device = find_gpu()


def train(
    train_dl: DataLoader,
    val_dl: DataLoader,
    view_value: dict[str, Any],
    eval_funcs: dict[str, Callable],
    **kv,
) -> Iterable[Sequence[dict[str, Any]]]:
    global model
    global args

    args.update(view_value)
    args.update({'device':device})

    def dict_dataloader_to_df(dataloader: DataLoader) -> pd.DataFrame:
        trajectory_id = 0
        trajectory_list, user_id_list, POI_id_list, region_id_list, time_period_list = (
            [],
            [],
            [],
            [],
            [],
        )

        for batch in dataloader:
            bs = len(next(iter(batch.values())))
            for i in range(bs):
                data_len = batch["mask"][i]
                trajectory_list.extend([trajectory_id] * data_len)
                user_id_list.extend([batch["user_id"][i].item()] * data_len)
                POI_id_list.extend(batch["POI_id"][i][:data_len].tolist())
                region_id_list.extend(batch["region_id"][i][:data_len].tolist())
                time_period_list.extend(batch["time_period"][i][:data_len].tolist())
                trajectory_id += 1

        records = pd.DataFrame(
            {
                "trajectory_id": trajectory_list,
                "user_id": user_id_list,
                "POI_id": POI_id_list,
                "region_id": region_id_list,
                "time_period": time_period_list,
            }
        )
        return records

    train_df = dict_dataloader_to_df(train_dl)
    graph_dict = construct_graph(args, train_df)

    model = iPCM(args, graph_dict)

    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        "max",
        factor=args.lr_scheduler_factor,
        patience=args.patience,
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        train_data_loader_tqdm = tqdm(train_dl, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            batch_data_device = {
                k: v.to(args.device, non_blocking=True)
                for k, v in batch_data.items()
                if "y_POI_id" not in k
            }
            batch_data_device["y_POI_id"] = {
                k: v.to(args.device, non_blocking=True)
                for k, v in batch_data["y_POI_id"].items()
            }
            loss = model(batch_data_device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() / len(batch_data)
            train_data_loader_tqdm.set_description(
                f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
            )

        log.info(f"Epoch: {epoch + 1}, train loss: {total_loss / len(train_dl)}")

        inference_res = inference(val_dl, view_value)
        y_predict = inference_res['pred']
        y_truth = inference_res['gts']
        scores = {}
        for name, func in eval_funcs.items():
            score = func(y_predict, y_truth)
            scores[name] = score

        lr_scheduler.step(scores["NDCG1"])

        yield [scores, {'loss': total_loss / len(train_dl), 'title':'train_loss'}]


def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kw
) -> Tuple[np.ndarray, np.ndarray]:
    assert model is not None

    model.eval()

    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(test_dl):
            batch_data_device = {
                k: v.to(model.device, non_blocking=True)
                for k, v in batch_data.items()
                if "y_POI_id" not in k
            }
            batch_data_device["y_POI_id"] = {
                k: v.to(model.device, non_blocking=True)
                for k, v in batch_data["y_POI_id"].items()
            }

            predict = model.predict(batch_data_device)
            y_predict_list.extend(predict.tolist())
            y_truth_list.extend(
                batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist()
            )

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)

    return {'pred':y_predict, 'gts':y_truth}
