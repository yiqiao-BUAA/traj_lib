# traj_lib/model/testmodel/main.py
import torch
from traj_lib.utils.logger import get_logger
log = get_logger(__name__)

def inference(dataloader, **kw):
    preds, gts = [], []
    for batch in dataloader:
        # toy example：恒预测 1
        preds.extend([1] * len(batch))
        gts.extend(batch.tolist())          # 假设 batch = tensor([...])
    log.debug("inference done, %d samples", len(preds))
    return preds, gts