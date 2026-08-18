"""
logger.py - nohup-safe rotating logger for G-MASS evaluation runs.
Owner: D | MediSafe-GH - Africa AI Safety Prize 2026
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("GMASS_LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "gmass_eval.log")
LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that writes to both console and a rotating log file.
    Safe to call multiple times with the same name - returns the same logger.

    Args:
        name: typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FMT, datefmt=DATE_FMT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning(f"Could not create log file at {LOG_FILE}. Logging to console only.")

    return logger
