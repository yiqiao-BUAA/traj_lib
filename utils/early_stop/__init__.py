# traj_lib/utils/dataloader/__init__.py
import importlib
import pkgutil
import sys
from utils.logger import get_logger
from utils.register import EARLY_STOP_REGISTRY

logger = get_logger(__name__)


def register_all(task=None):
    logger.info(f"Registering all Early Stop for task: {task}")
    for mod in pkgutil.iter_modules([f"{__path__[0]}/{task}"], prefix=f"{__name__}."):
        before = set(EARLY_STOP_REGISTRY)
        try:
            importlib.import_module(
                f'{".".join(mod.name.split(".")[:-1])}.{task}.{mod.name.split(".")[-1]}'
            )
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            added = set(EARLY_STOP_REGISTRY) - before
            for k in added:
                try:
                    EARLY_STOP_REGISTRY.pop(k, None)
                except Exception:
                    pass

            sys.modules.pop(mod.name, None)

            logger.error(f"Failed to import {mod.name}: {e}", exc_info=True)
            continue

        new_keys = set(EARLY_STOP_REGISTRY) - before
        for key in new_keys:
            logger.debug(f"Early Stop {key:>15s} found from {mod.name}")
