# traj_lib/utils/eval/accuracy.py
from traj_lib.utils.register import register_eval
import numpy as np

@register_eval("accuracy")
def evaluate(preds, gts):
    """
    基础分类准确率
    preds / gts: list[int] 或 ndarray
    return: float in [0,1]
    """
    preds = np.asarray(preds)
    gts   = np.asarray(gts)
    assert preds.shape == gts.shape
    return (preds == gts).mean()
