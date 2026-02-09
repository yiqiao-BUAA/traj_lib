from typing import Sequence, Tuple, Optional, Callable, Any
from collections.abc import Iterable

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from .MTNet import MTNet


args = ConfigResolver("./model/MTNet/MTNet.yaml").parse()
log = get_logger(__name__)

pre_views = ["MTNet_preview"]
post_views = ["MTNet_post_view"]

model: Optional[MTNet] = None


def train(
    train_dl: DataLoader,
    val_dl: DataLoader,
    view_value: dict[str, Any],
    eval_funcs: dict[str, Callable],
    **kw,
) -> Iterable[Sequence[dict[str, Any]]]:

    global model
    global args

    args.update(view_value)

    model = MTNet(args)

    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=args.lr_step_size, gamma=args.lr_gamma
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
        lr_scheduler.step(epoch)

        y_predict, y_truth = inference(val_dl, view_value)
        scores = {}
        for name, func in eval_funcs.items():
            score = func(y_predict, y_truth)
            scores[name] = score

        yield [scores, {'loss': total_loss / len(train_dl), 'title': 'train_loss'}]


def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kw
) -> Tuple[np.ndarray, np.ndarray]:
    assert model is not None

    log.info("=============begin valid/test==============")
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
            y_predict_list.extend(predict.detach().cpu().numpy().tolist())
            y_truth_list.extend(
                batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist()
            )

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)

    return y_predict, y_truth
