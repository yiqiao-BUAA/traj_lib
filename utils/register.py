# traj_lib/utils/register.py

from collections import defaultdict
from typing import Callable, Dict

_REGISTRY: Dict[str, Dict[str, object]] = defaultdict(dict)

def _register(category: str, name: str) -> Callable[[object], object]:
    def decorator(obj: object) -> object:
        reg = _REGISTRY[category]
        if name in reg:
            raise KeyError(f"{category} '{name}' already registered")
        reg[name] = obj
        return obj
    return decorator


def register_dataset(name: str):
    return _register("dataloader", name)

def register_eval(name: str):
    return _register("eval", name)


@property
def DATASET_REGISTRY() -> Dict[str, object]:      # noqa: N802
    return _REGISTRY["dataloader"]

@property
def EVAL_REGISTRY() -> Dict[str, object]:         # noqa: N802
    return _REGISTRY["eval"]

DATALOADER_REGISTRY     = _REGISTRY["dataloader"]
EVAL_REGISTRY           = _REGISTRY["eval"]