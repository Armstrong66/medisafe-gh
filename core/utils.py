r"""
core.utils — Shared I/O, caching, and environment helpers.
Owner: D (Engineering Lead)  |  MediSafe-GH · Africa AI Safety Prize 2026

Unified from two parallel implementations (Team D scratch work + the
GMASS_Coding_Standard.md reference repo). Function names from BOTH
versions are kept as aliases so nothing else in the codebase breaks:

    load_jsonl()         — returns [] on missing file (does not raise)
    append_jsonl()        \__ same function, two names
    save_jsonl_line()     /
    load_completed_ids()  — works with either function name above

Key cost-saving utility: load_completed_ids() enables crash-safe resumption
of API batches — never re-pay for a probe already evaluated.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# JSONL I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_jsonl(path: str | Path, warn_missing: bool = True) -> list[dict]:
    """
    Load all records from a JSONL file.

    Returns [] if the file is missing (does not raise) — this matches the
    coding-standard reference implementation. Batch scripts can call this
    on an output file that doesn't exist yet without wrapping in try/except.
    """
    p = Path(path)
    if not p.exists():
        message = f"JSONL not found: {p} - returning []"
        if warn_missing:
            logger.warning(message)
        else:
            logger.debug(message)
        return []

    records, errors = [], 0
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"JSON error line {i} of {p}: {e}")
                errors += 1

    if errors:
        logger.warning(f"Loaded {len(records)} records with {errors} parse errors from {p}")
    else:
        logger.debug(f"Loaded {len(records)} records from {p}")
    return records


def append_jsonl(record: dict, path: str | Path) -> None:
    """
    Append ONE record to a JSONL file (creates file + parent dirs if missing).

    Always append — never overwrite — during batch runs. A crashed run loses
    at most one in-flight record, not the entire batch.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Alias — earlier pipeline.py / scorer.py scripts call this name.
# Keeping both names means neither version of the codebase needs editing.
save_jsonl_line = append_jsonl


def load_completed_ids(output_path: str | Path, id_field: str = "probe_id") -> set[str]:
    """
    Return the set of probe_ids already present in an output JSONL.

    Use this at the start of every batch run to skip already-evaluated probes.
    This is the primary API cost-saving mechanism: zero re-calls on resume.

    Example:
        done   = load_completed_ids("data/eval_outputs/raw/gpt-4o.jsonl")
        probes = [p for p in all_probes if p["probe_id"] not in done]
        logger.info(f"Resuming: {len(probes)} probes remaining")
    """
    records = load_jsonl(output_path, warn_missing=False)
    ids = {r[id_field] for r in records if id_field in r}
    if ids:
        logger.info(f"Resume: {len(ids)} probes already done in {Path(output_path).name}")
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def is_kaggle() -> bool:
    """True when running inside a Kaggle kernel."""
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None


def is_cuda_available() -> bool:
    """True when a CUDA GPU (RTX) is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """Return 'cuda' on RTX, else 'cpu'. Used by local model loaders (LlamaGuard3, RoBERTa)."""
    return "cuda" if is_cuda_available() else "cpu"


def log_environment(logger_instance) -> None:
    """Log a one-line environment summary at run start."""
    if is_kaggle():
        env = "Kaggle (T4 GPU)" if is_cuda_available() else "Kaggle (CPU)"
    elif is_cuda_available():
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            env  = f"Local CUDA — {name}"
        except Exception:
            env = "Local CUDA"
    else:
        env = "CPU only"
    logger_instance.info(f"Environment: {env}")


# ══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_api_key(env_var: str) -> str:
    """
    Retrieve API key from environment. Raises a clear ValueError if missing.
    Never hardcode keys in scripts — always use .env + dotenv.
    """
    key = os.getenv(env_var)
    if not key:
        raise ValueError(
            f"'{env_var}' not set. Add it to your .env file and call load_dotenv()."
        )
    return key


def retry_with_backoff(fn, retries: int = 3, base_wait: float = 1.0):
    """
    Call fn() with exponential backoff on RateLimitError.
    Returns None on final failure — never crashes the batch.

    Args:
        fn        : zero-argument callable (lambda wrapping the API call)
        retries   : max attempts
        base_wait : initial wait in seconds (doubles each retry)
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            err_str = str(e).lower()
            is_rate = any(x in err_str for x in ["rate limit", "429", "quota"])
            wait = base_wait * (2 ** attempt)
            if is_rate:
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{retries}). Waiting {wait}s.")
                time.sleep(wait)
            else:
                logger.error(f"API error (attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    return None
                time.sleep(wait)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MISC
# ══════════════════════════════════════════════════════════════════════════════

def utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs(*paths: str) -> None:
    """Create one or more directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Directory ensured: {path}")


def validate_probe_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitise probe text before sending to model APIs (GMASS Security Engineering §9).
    Removes ASCII control characters, limits length, and flags prompt injection patterns.
    """
    if not isinstance(text, str):
        text = str(text or "")
    # Remove control characters except tab and newline
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    if len(sanitized) > max_length:
        raise ValueError(f"Probe text exceeds maximum allowed length: {len(sanitized)} > {max_length} chars")

    injection_patterns = [
        r"ignore (?:all )?previous instructions",
        r"you are now",
        r"disregard (?:all )?prior",
        r"system prompt",
        r"jailbreak",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Potential injection pattern flagged in probe input: '{pattern}'")

    return sanitized

