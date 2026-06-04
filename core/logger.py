"""
medisafe_gh.core.logger — shared logging helper.

Supports interactive sessions (Kaggle, Jupyter), terminal runs,
and nohup background runs on RTX without disturbing downstream code.

Usage:
    from medisafe_gh.core.logger import get_logger
    logger = get_logger(__name__)

nohup pattern (RTX):
    nohup python -m medisafe_gh.cli evaluate > logs/nohup_out.txt 2>&1 &
    tail -f logs/gmass_eval.log
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR   = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE  = LOG_DIR / "gmass_eval.log"
MAX_BYTES = 5 * 1024 * 1024   # 5 MB — rotate after this
BACKUPS   = 3
LEVEL     = os.getenv("GMASS_LOG_LEVEL", "INFO").upper()

# ── Internal helpers ────────────────────────────────────────────────────────────────
def _is_nohup() -> bool:
    return not sys.stdout.isatty()


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _plain_formatter() -> logging.Formatter:
    return logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _file_handler() -> RotatingFileHandler:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(LOG_FILE, maxBytes=MAX_BYTES,
                            backupCount=BACKUPS, encoding="utf-8")
    h.setFormatter(_plain_formatter())
    h.setLevel(LEVEL)
    return h


def _console_handler() -> logging.Handler:
    if _is_nohup():
        # Detached run — plain stderr only (stdout → nohup_out.txt)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(_plain_formatter())
        h.setLevel(LEVEL)
        return h
    try:
        from rich.logging import RichHandler
        h = RichHandler(rich_tracebacks=True, markup=True, show_path=False)
        h.setLevel(LEVEL)
        return h
    except ImportError:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_plain_formatter())
        h.setLevel(LEVEL)
        return h

# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Always writes to rotating log file + console."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(LEVEL)
    logger.addHandler(_file_handler())
    logger.addHandler(_console_handler())
    logger.propagate = False
    return logger


def log_run_header(logger: logging.Logger, config: dict) -> None:
    """Structured header logged at the start of every evaluation run."""
    sep = "=" * 64
    logger.info(sep)
    logger.info(f"G-MASS RUN START — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in config.items():
        logger.info(f"  {k:<22}: {v}")
    logger.info(sep)


def log_run_footer(logger: logging.Logger, n_done: int,
                   n_total: int, elapsed: float) -> None:
    """Structured footer logged at the end of every evaluation run."""
    sep = "=" * 64
    logger.info(sep)
    logger.info("G-MASS RUN COMPLETE")
    logger.info(f"  Completed : {n_done}/{n_total}")
    logger.info(f"  Elapsed   : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Log file  : {LOG_FILE}")
    logger.info(sep)