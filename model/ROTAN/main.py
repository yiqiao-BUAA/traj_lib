from typing import Tuple, Callable, Any, Sequence
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from .ROTAN import ROTAN

args = ConfigResolver("./model/ROTAN/ROTAN.yaml").parse()
log = get_logger(__name__)

pre_views = ["ROTAN_preview", "analyze_POI"]
post_views = ["ROTAN_postview"]

model: ROTAN | None = None

torch.set_num_threads(1)


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

    model = ROTAN(args)

    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    lr_scheduler = CosineLRScheduler(
        optimizer=opt,
        t_initial=args.num_epochs,
        lr_min=1e-5,
        warmup_t=10,
        warmup_lr_init=args.warmup_lr_init,
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_data_loader_tqdm = tqdm(train_dl, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            batch_data["quad_key"] = torch.Tensor(batch_data["quad_key"])
            batch_data["y_POI_id"]["quad_key"] = torch.Tensor(
                batch_data["y_POI_id"]["quad_key"]
            )
            batch_data_device = {
                k: v.to(model.device, non_blocking=True)
                for k, v in batch_data.items()
                if "y_POI_id" not in k
            }
            batch_data_device["y_POI_id"] = {
                k: v.to(model.device, non_blocking=True)
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
        # log.info(f"Epoch: {epoch + 1}, gate values: {model.get_gate_values()}")
        lr_scheduler.step(epoch)
    
        results = inference(val_dl, view_value)
        scores = {}
        for name, func in eval_funcs.items():
            score = func(results['pred'], results['gts'])
            scores[name] = score

        yield [scores, {'loss': total_loss / len(train_dl), 'title':'train loss'}]


def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kw
) -> Tuple[np.ndarray, np.ndarray]:
    assert model is not None

    log.info("=============begin valid/test==============")
    model.eval()
    user_list = []
    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        mask_list = []
        for batch_data in tqdm(test_dl):
            batch_data["quad_key"] = torch.Tensor(batch_data["quad_key"])
            batch_data["y_POI_id"]["quad_key"] = torch.Tensor(
                batch_data["y_POI_id"]["quad_key"]
            )

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
            mask_list.extend(
                batch_data["mask"].detach().cpu().numpy().tolist()
            )
            user_list.extend(
                batch_data["user_id"].detach().cpu().numpy().tolist()
            )

    y_predict = np.array(y_predict_list)
    y_truth = np.array(y_truth_list)


    # culculate the correct rate based by bucket
    y_predict_num = np.argmax(y_predict, axis=-1)
    bucket_correct_dict = {}
    bucket_total_dict = {}
    poi_buckets = view_value['poi_buckets']
    for truth, predict in zip(y_truth.flatten(), y_predict_num.flatten()):
        for bucket_name, poi_list in poi_buckets.items():
            if truth in poi_list:
                bucket_total_dict[bucket_name] = bucket_total_dict.get(bucket_name, 0) + 1
                if truth == predict:
                    bucket_correct_dict[bucket_name] = bucket_correct_dict.get(bucket_name, 0) + 1
                break
    for bucket_name in poi_buckets.keys():
        correct = bucket_correct_dict.get(bucket_name, 0)
        total = bucket_total_dict.get(bucket_name, 1)
        accuracy = correct / total
        log.info(f"Bucket: {bucket_name}, Accuracy: {accuracy:.4f} ({correct}/{total})")


    return {'pred': y_predict, 'gts': y_truth}
