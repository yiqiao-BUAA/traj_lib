# traj_lib/utils/logger.py
from __future__ import annotations

import logging
import warnings
from logging import FileHandler, StreamHandler
from pathlib import Path
from datetime import datetime
import os

# ---------- 路径与文件名 ----------
ROOT_LOGGER_NAME = "traj_lib"                          # 项目根 logger
ROOT_DIR = Path(__file__).resolve().parent.parent      # traj_lib/
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = LOG_DIR / f"benchmark-{_ts}-{os.getpid()}.log"

# ---------- 根 logger 只配置一次 ----------
def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:          # 已配置过则直接复用
        return root

    root.setLevel(logging.DEBUG)

    # 1) 终端输出（INFO+）
    console = StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    # 2) 文件输出（DEBUG+，完整保留）
    file = FileHandler(LOG_FILE, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"))

    root.addHandler(console)
    root.addHandler(file)

    # ---------- 把 warnings 也写进日志 ----------
    logging.captureWarnings(True)      # warnings.warn → py.warnings → handler
    warnings.filterwarnings("default") # 显示同类 warning，不合并

    return root


# ---------- 对外统一入口 ----------
def get_logger(name: str | None = None,
               level: int = logging.DEBUG) -> logging.Logger:
    """
    任何模块请调用：
        log = get_logger(__name__)
    子 logger 不再单独挂 handler，只设级别并保持 propagate=True。
    """
    _configure_root_logger()
    lg = logging.getLogger(name or ROOT_LOGGER_NAME)
    lg.setLevel(level)
    lg.propagate = True
    return lg
