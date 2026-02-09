#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
from utils.logger import get_logger

logger = get_logger(__name__)

# Maintain at the top: any 'cuda:x' in this list will be excluded from selection.
avoid_gpu = [
    # 'cuda:0',
    # 'cuda:1',
    # 'cuda:2'
]


def _avoid_gpu_indices():
    """Parse avoid_gpu like ['cuda:0', 'cuda:2'] into a set of integer indices: {0, 2}."""
    indices = set()
    for s in avoid_gpu:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        if not s.startswith("cuda:"):
            continue
        try:
            indices.add(int(s.split(":", 1)[1]))
        except (ValueError, IndexError):
            continue
    return indices


def _filter_avoided_gpus(free_memories):
    """Remove entries whose gpu index is in avoid_gpu."""
    avoided = _avoid_gpu_indices()
    if not avoided:
        return free_memories
    filtered = [(idx, free) for (idx, free) in free_memories if idx not in avoided]
    if len(filtered) != len(free_memories):
        logger.debug(f"Filtered avoided GPUs: {sorted(avoided)}. Remaining candidates: {[i for i, _ in filtered]}")
    return filtered


def get_gpu_free_by_pynvml():
    try:
        import pynvml
    except ImportError:
        logger.info('No pynvml installed, will not use pynvml to get GPU info. Maybe install it via "pip install pynvml".')
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        free_memories = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memories.append((i, info.free))

        return free_memories
    except Exception as e:
        logger.warning(f"Failed to get GPU info, error: {e}")
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_gpu_free_by_nvidia_smi():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.error(f"Unable to run nvidia-smi to get GPU info, error: {e}")
        return None

    free_memories = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        index_str, free_mb_str = parts
        try:
            idx = int(index_str)
            free_mb = int(free_mb_str)
            free_bytes = free_mb * 1024 * 1024
            free_memories.append((idx, free_bytes))
        except ValueError:
            continue

    return free_memories


def pick_best_gpu(free_memories):
    return max(free_memories, key=lambda x: x[1])


def find_gpu():
    free_memories = get_gpu_free_by_pynvml()

    if free_memories is None:
        free_memories = get_gpu_free_by_nvidia_smi()

    if not free_memories:
        logger.error("No available NVIDIA GPU detected, or pynvml / nvidia-smi not usable.")
        raise RuntimeError("No available NVIDIA GPU detected.")

    # Filter out avoided GPUs
    free_memories = _filter_avoided_gpus(free_memories)

    if not free_memories:
        logger.error(f"All GPUs are excluded by avoid_gpu={avoid_gpu}, or no GPUs left after filtering.")
        raise RuntimeError("No available NVIDIA GPU detected (all excluded by avoid_gpu).")

    best_idx, best_free = pick_best_gpu(free_memories)

    logger.debug(f"Selected GPU index: {best_idx}, free memory: {best_free / (1024**3):.2f} GB.")
    return f"cuda:{best_idx}"
