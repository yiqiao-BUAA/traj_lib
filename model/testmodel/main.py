# traj_lib/model/testmodel/main.py

import torch

from traj_lib.utils.logger import get_logger
log = get_logger(__name__)

def inference(dataloader, **kw):
    preds, gts = [], []
    total_steps = len(dataloader)
    log.info("inference started, %d steps", total_steps)
    for batch in dataloader:
        preds.extend(batch['POI_id'][:, -1])
        gts.extend(batch['y_POI_id'])
        log.info("inference done, %d samples %d output", len(preds), len(gts))
    preds = torch.tensor(preds, dtype=torch.long)
    gts = torch.tensor(gts, dtype=torch.long)
    return preds, gts