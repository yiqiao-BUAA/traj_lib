# traj_lib/utils/dataloader/__init__.py
import importlib, pkgutil
from traj_lib.utils.logger import get_logger
from traj_lib.utils.register import DATALOADER_REGISTRY

logger = get_logger(__name__)

for mod in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
    before = set(DATALOADER_REGISTRY)
    importlib.import_module(mod.name)
    new_keys = set(DATALOADER_REGISTRY) - before
    for key in new_keys:
        logger.debug(f"DataLoader {key} found from {__name__}")
