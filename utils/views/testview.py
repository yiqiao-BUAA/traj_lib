import pandas as pd
from torch.utils.data import Dataset

from traj_lib.utils.register import register_view
from traj_lib.utils.logger import get_logger

logger = get_logger(__name__)

@register_view("testview1")
def testview1(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A simple view that returns the last POI_id and y_POI_id from the dataset.
    """
    logger.info("Applying testview1 to dataset")
    view_value['testview'] = 1
    return raw_df, view_value

@register_view("testview2")
def testview2(raw_df: pd.DataFrame, view_value: dict = None) -> tuple[Dataset, dict]:
    """
    A simple view that returns the last POI_id and y_POI_id from the dataset.
    """
    logger.info("Applying testview2 to dataset")
    raw_df['testview2'] = raw_df['user_id'] + raw_df['POI_id']
    return raw_df, view_value