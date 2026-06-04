"""
medisafe_gh — G-MASS: Ghana Medical AI Safety Screen.

A cross-lingual safety evaluation suite for medical AI in Ghanaian languages.
Africa AI Safety Prize Competition 2026 · Track II.

Quick start:
    from medisafe_gh.core.metrics import full_model_profile
    from medisafe_gh.scoring.scorer import GMassScorer
    from medisafe_gh.probes.loader import load_probes

Package layout:
    medisafe_gh/
    ├── core/           — metrics (CSR, SDS, RAR), config, logging, utils
    ├── scoring/        — LlamaGuard3 + RoBERTa pipeline, referral detector
    ├── probes/         — probe loader, builder, translation helpers
    ├── audio/          — Whisper ASR + Khaya TTS helpers
    └── cli.py          — `gmass` command-line entry point
"""

__version__ = "0.1.0"
__author__  = "MediSafe-GH Team"

# Surface the most-used names at top level for convenience
from medisafe_gh.core.metrics import (       # noqa: F401
    compute_csr,
    compute_sds,
    compute_rar,
    full_model_profile,
)