"""medisafe_gh.core.config — load and validate YAML configs."""

from pathlib import Path
import yaml
from medisafe_gh.core.logger import get_logger

logger = get_logger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_config(name: str) -> dict:
    """
    Load a YAML config by filename (without extension).
    Looks in the repo-level configs/ directory.

    Example:
        cfg = load_config("gmass_config")
        cfg = load_config("models")
    """
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config '{name}.yaml' not found in {_CONFIG_DIR}. "
            "Check spelling or ensure configs/ directory is present."
        )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Loaded config: {path.name}")
    return cfg