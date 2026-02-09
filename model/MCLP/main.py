import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Optional, Callable, Any, Sequence
from collections.abc import Iterable

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from tqdm import tqdm

from .MCLP import MCLP
from transformers import get_linear_schedule_with_warmup

args = ConfigResolver("./model/MCLP/MCLP.yaml").parse()
log = get_logger(__name__)

pre_views = ["MCLP_preview"]
post_views = ["MCLP_post_view"]

model: Optional[MCLP] = None


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

    model = MCLP(args)
    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    warmup_scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=len(train_dl) * 1,
        num_training_steps=len(train_dl) * args.num_epochs,
    )

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( opt, T_max=args.num_epochs, eta_min=1e-6)
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_data_loader_tqdm = tqdm(train_dl, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            batch_data_device = {
                k: (
                    torch.as_tensor(v, device=args.device)
                    if isinstance(v, (list, np.ndarray))
                    else v.to(args.device, non_blocking=True)
                )
                for k, v in batch_data.items()
                if "y_POI_id" not in k
            }
            batch_data_device["y_POI_id"] = {
                k: (
                    torch.as_tensor(v, device=args.device)
                    if isinstance(v, (list, np.ndarray))
                    else v.to(args.device, non_blocking=True)
                )
                for k, v in batch_data["y_POI_id"].items()
            }

            loss = model.calculate_loss(batch_data_device)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            warmup_scheduler.step()

            total_loss += loss.item() / len(batch_data)

            train_data_loader_tqdm.set_description(
                f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
            )

        log.info(f"Epoch: {epoch + 1}, train loss: {total_loss / len(train_dl)}")

        tmp_res = inference(val_dl, view_value)
        y_predict, y_truth = tmp_res['pred'], tmp_res['gts']
        scores = {}
        for name, func in eval_funcs.items():
            score = func(y_predict, y_truth)
            scores[name] = score

        yield [scores, {"loss": total_loss / len(train_dl), "title": "train_loss"}]


def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kw
) -> Tuple[np.ndarray, np.ndarray]:
    assert model is not None

    model.eval()

    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(test_dl):
            batch_data_device = {
                k: (
                    torch.as_tensor(v, device=model.device)
                    if isinstance(v, (list, np.ndarray))
                    else v.to(model.device, non_blocking=True)
                )
                for k, v in batch_data.items()
                if "y_POI_id" not in k
            }
            batch_data_device["y_POI_id"] = {
                k: (
                    torch.as_tensor(v, device=model.device)
                    if isinstance(v, (list, np.ndarray))
                    else v.to(model.device, non_blocking=True)
                )
                for k, v in batch_data["y_POI_id"].items()
            }

            predict = model.predict(batch_data_device)

            y_predict_list.extend(
                predict.detach()
                .cpu()
                .numpy()
                .reshape(-1, predict.shape[-1])
                .tolist()
            )
            y_truth_list.extend(
                batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist()
            )

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)
    res = {'pred': y_predict, 'gts': y_truth}

    return res
