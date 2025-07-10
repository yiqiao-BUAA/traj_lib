# traj_lib/utils/logger.py
import logging
from logging import FileHandler, StreamHandler
from pathlib import Path
from datetime import datetime
import os

ROOT_LOGGER_NAME = "traj_lib"                    # 你自己的根
ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR  = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

_run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = LOG_DIR / f"benchmark-{_run_ts}-{os.getpid()}.log"


def _configure_root_logger():
    """只在第一次调用时为根 logger 加 handler"""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.handlers:                 # 已经配置过就什么也不做
        return root

    root.setLevel(logging.DEBUG)

    console = StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    file = FileHandler(LOG_FILE, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"))

    root.addHandler(console)
    root.addHandler(file)
    return root


def get_logger(name: str | None = None, level: int = logging.DEBUG):
    """
    统一入口：任何模块都 `get_logger(__name__)`
    只有根 logger 挂 handler，子 logger 复用即可
    """
    _configure_root_logger()
    lg = logging.getLogger(name or ROOT_LOGGER_NAME)
    lg.setLevel(level)                # 仅设级别，不再加 handler
    return lg
