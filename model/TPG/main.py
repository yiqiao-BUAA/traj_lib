from typing import Tuple, Optional, Callable, Any, Sequence
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger
from utils.exargs import ConfigResolver
from .TPG import TPG

args = ConfigResolver("./model/TPG/TPG.yaml").parse()
log = get_logger(__name__)

pre_views = ["TPG_preview", "analyze_POI"]
post_views = ["TPG_post_view"]

model: Optional[TPG] = None


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

    model = TPG(args)

    model = model.to(args.device)

    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        train_data_loader_tqdm = tqdm(train_dl, ncols=120)
        for batch_idx, batch_data in enumerate(train_data_loader_tqdm):
            loss = model.calculate_loss(batch_data)

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


        yield [scores, {'loss': total_loss / len(train_dl), 'title': 'train_loss'}]


# ===================== Inference =====================
def inference(
    test_dl: DataLoader, view_value: dict[str, Any], **kw
) -> Tuple[np.ndarray, np.ndarray]:
    assert model is not None
    log.info("=============begin valid/test==============")
    model.eval()

    with torch.no_grad():
        y_predict_list, y_truth_list = [], []
        for batch_data in tqdm(test_dl):
            predict = model.predict(batch_data)
            y_predict_list.extend(predict.detach().cpu().numpy().tolist())
            y_truth_list.extend(
                batch_data["y_POI_id"]["POI_id"].detach().cpu().numpy().tolist()
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
