# traj_lib/data/__init__.py
import importlib, pkgutil
from traj_lib.utils.logger import get_logger
logger = get_logger(__name__)

for mod in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{mod.name}")
    logger.debug(f"DataLoader {mod.name} found from {__name__}")
