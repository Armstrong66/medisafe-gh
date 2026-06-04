"""
logger.py — G-MASS shared logging helper.

Supports:
  - Interactive sessions (Kaggle, Jupyter, terminal)
  - nohup background runs on RTX (log to file, no terminal dependency)
  - Rotating file handler so logs never grow unbounded
  - Rich console output when running interactively, plain text for nohup

Usage:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting evaluation batch")

For nohup runs (RTX):
    nohup python -m src.evaluate --config configs/models.yaml > logs/nohup_out.txt 2>&1 &
    tail -f logs/gmass_eval.log   # live-follow the structured log

Owner: A (Team Lead)
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR   = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE  = LOG_DIR / "gmass_eval.log"
MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUPS   = 3                  # keep gmass_eval.log + 3 rotated backups
LOG_LEVEL = os.getenv("GMASS_LOG_LEVEL", "INFO").upper()

# ── Internal helpers ──────────────────────────────────────────────────────────
def _is_nohup() -> bool:
    """Detect whether we are running detached (nohup / no tty)."""
    return not sys.stdout.isatty()


def _is_notebook() -> bool:
    """Detect Jupyter / Kaggle notebook environment."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _make_formatter(plain: bool = False) -> logging.Formatter:
    if plain:
        # Plain text — safe for nohup log files and CI
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    else:
        # Same format — rich colours applied at handler level if available
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        return logging.Formatter(fmt, datefmt="%H:%M:%S")


def _file_handler() -> RotatingFileHandler:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(_make_formatter(plain=True))
    handler.setLevel(LOG_LEVEL)
    return handler


def _console_handler() -> logging.Handler:
    """
    Returns a Rich handler when available and running interactively,
    otherwise a plain StreamHandler (safe for nohup).
    """
    if _is_nohup():
        # nohup / background run — write plain text to stderr only
        # (stdout is typically redirected to nohup_out.txt by the user)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_make_formatter(plain=True))
        handler.setLevel(LOG_LEVEL)
        return handler

    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        handler.setLevel(LOG_LEVEL)
        return handler
    except ImportError:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_make_formatter(plain=False))
        handler.setLevel(LOG_LEVEL)
        return handler


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Always writes to:
      - logs/gmass_eval.log    (rotating file, always on)
      - stderr/stdout console  (rich when interactive, plain when nohup)

    Args:
        name: typically __name__ of the calling module.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)
    logger.addHandler(_file_handler())
    logger.addHandler(_console_handler())
    logger.propagate = False   # don't bubble to root logger
    return logger


def log_run_header(logger: logging.Logger, config: dict) -> None:
    """
    Log a structured header at the start of any evaluation run.
    Makes it easy to identify runs when reviewing log files after nohup.

    Args:
        logger: logger instance from get_logger().
        config: dict of run parameters (model, language, n_probes, etc.)
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info(f"G-MASS EVALUATION RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for key, val in config.items():
        logger.info(f"  {key:<20}: {val}")
    logger.info(sep)


def log_run_footer(logger: logging.Logger, n_completed: int,
                   n_total: int, elapsed_sec: float) -> None:
    """
    Log a structured footer at the end of any evaluation run.

    Args:
        logger:      logger instance.
        n_completed: number of probes successfully scored.
        n_total:     total probes attempted.
        elapsed_sec: wall-clock seconds for the run.
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("G-MASS RUN COMPLETE")
    logger.info(f"  Completed : {n_completed} / {n_total} probes")
    logger.info(f"  Elapsed   : {elapsed_sec:.1f}s  ({elapsed_sec/60:.1f} min)")
    logger.info(f"  Log file  : {LOG_FILE}")
    logger.info(sep)