# utils/registry.py
DATALOADER_REGISTRY = {}

def register_dataset(name):
    def decorator(fn):
        DATALOADER_REGISTRY[name] = fn
        return fn
    return decorator
