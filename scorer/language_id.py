"""
scorer.language_id — Response-language detection via fasttext.
Owner: D (Engineering Lead)  |  MediSafe-GH · Africa AI Safety Prize 2026

WHY THIS MODULE EXISTS
-----------------------
The scorer must route each model RESPONSE to the right primary scorer:
    Twi response     → AfroLM primary, LlamaGuard3 (via Khaya) secondary
    English/GH-EN     → LlamaGuard3 primary, AfroLM secondary

The probe's declared `language` field ("twi", "english", "ghanaian_en") is
NOT a reliable signal for this decision. Per GMASS_Team_Clarifications.md
§8, frontier models frequently answer in English even when prompted in
Twi. Routing on the probe's declared language — instead of what the model
actually wrote — silently sends English text to AfroLM as primary judge,
the exact scenario §8 already warned the team to expect.

Two earlier candidate fixes were considered and rejected:
  1. ASCII / Twi-diacritic density check (ɛ, ɔ, ŋ) — distinguishes Twi from
     English only weakly, cannot distinguish GH-EN from standard EN, and
     cannot distinguish Twi from other diacritic-using languages (e.g. Ewe).
  2. A small LLM call for language ID — solves a microsecond problem with
     a multi-second, API-cost-bearing tool. Unnecessary overhead.

fasttext's lid.176 model is the standard NLP-community tool for exactly
this: ~1MB (compressed .ftz variant), covers 176 languages including Twi
(__label__tw), runs in microseconds, needs no GPU. This module wraps it
as the single source of truth for "what language is this text actually
in," used both for scorer routing AND for the §8 accessibility finding
(see LanguageConsistencyChecker in scorer.py).

This project only distinguishes twi vs. not-twi today. Anything fasttext
detects that isn't Twi collapses to "english" — this deliberately covers
standard EN, GH-EN, and any other Latin-script answer, matching the
clarifications doc's own routing table, which treats EN and GH-EN
identically ("RoBERTa/AfroLM + LG3 for EN and GH-EN").

Usage:
    from scorer.language_id import detect_response_language

    lang = detect_response_language(model_response_text)
    # lang is one of: "twi", "english"
"""

import os
import urllib.request
from pathlib import Path
from core.logger import get_logger

logger = get_logger(__name__)

# fasttext label → our internal language taxonomy.
_FASTTEXT_TO_INTERNAL = {
    "tw": "twi",
}

# Where the model file lives once downloaded. Kept small (.ftz, ~1MB)
# rather than the full lid.176.bin (~126MB) — same 176-language coverage,
# compressed via fasttext's built-in quantization.
DEFAULT_MODEL_DIR  = Path(os.getenv(
    "FASTTEXT_LID_MODEL_DIR",
    str(Path(__file__).parent / "models"),
))
DEFAULT_MODEL_PATH = Path(os.getenv(
    "FASTTEXT_LID_MODEL_PATH",
    str(DEFAULT_MODEL_DIR / "lid.176.ftz"),
))
MODEL_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

# Confidence below this → treat the prediction as unreliable and fall back
# to "english". English-track scoring (LlamaGuard3 primary, AfroLM
# secondary) is the safer default for an uncertain case, since LlamaGuard3
# is the more broadly validated of the two scorers.
MIN_CONFIDENCE = 0.40

_model = None  # lazy-loaded singleton — load once per process, not per call


def _ensure_model_downloaded() -> Path:
    """
    Return the path to a usable lid.176.ftz, downloading it if missing.

    Raises:
        RuntimeError if the file is absent AND the download fails. There is
        no safe silent fallback here — routing on the wrong signal is the
        exact bug this module exists to fix, so we fail loudly rather than
        let scorer routing run with no language signal at all.
    """
    if DEFAULT_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH

    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"fasttext LID model not found at {DEFAULT_MODEL_PATH} — downloading")
    try:
        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, DEFAULT_MODEL_PATH)
        logger.info(f"Downloaded fasttext LID model to {DEFAULT_MODEL_PATH}")
        return DEFAULT_MODEL_PATH
    except Exception as e:
        raise RuntimeError(
            f"Could not download fasttext LID model from {MODEL_DOWNLOAD_URL}: {e}\n"
            f"Download manually and place at {DEFAULT_MODEL_PATH}, "
            f"or set FASTTEXT_LID_MODEL_PATH in .env to an existing copy.\n"
            f"This model is REQUIRED for scorer routing — there is no safe "
            f"fallback, since routing on the wrong signal is the exact bug "
            f"this module fixes."
        ) from e


def _load_model():
    """Lazy-load the fasttext LID model singleton."""
    global _model
    if _model is not None:
        return _model

    import fasttext

    path = _ensure_model_downloaded()

    # fasttext prints a harmless deprecation warning on load; suppress noise.
    fasttext.FastText.eprint = lambda *args, **kwargs: None
    _model = fasttext.load_model(str(path))
    logger.info(f"fasttext LID model loaded from {path}")
    return _model


def detect_response_language(text: str) -> str:
    """
    Detect the actual language of a model's response text.

    Args:
        text : the model's response (any language)

    Returns:
        "twi"     — fasttext confidently predicts Twi (__label__tw)
        "english" — everything else: empty/whitespace text (short-circuits
                    before any model load — common case for API-error
                    responses, no need to pay a model load for it), any
                    non-Twi fasttext prediction (covers EN, GH-EN, and any
                    other language — see module docstring), or a low-
                    confidence Twi guess (MIN_CONFIDENCE gate)

    This function is the ONLY place scorer-routing language decisions
    should originate from. Do not re-implement a diacritic check or any
    other ad-hoc heuristic elsewhere in the pipeline — that duplication is
    exactly how the original probe-language-routing bug went unnoticed.
    """
    result, _confidence = detect_response_language_with_confidence(text)
    return result


def detect_response_language_with_confidence(text: str) -> tuple[str, float]:
    """
    Same as detect_response_language(), but also returns fasttext's raw
    confidence score. Useful for logging/debugging routing decisions and
    for the §8 LanguageConsistencyChecker, which may want to log how
    confident a "this looks like English" call actually was.

    Returns:
        (language, confidence) where language is "twi" or "english".
        Empty/whitespace text returns ("english", 1.0) — fully confident
        by definition, since there's nothing to misclassify.
    """
    cleaned = (text or "").replace("\n", " ").strip()
    if not cleaned:
        logger.debug("detect_response_language called with empty text — returning 'english'")
        return "english", 1.0

    model = _load_model()
    labels, confidences = _predict_one(model, cleaned)

    label      = labels[0].replace("__label__", "")
    confidence = float(confidences[0])

    if label == "tw" and confidence < MIN_CONFIDENCE:
        logger.debug(
            f"fasttext low-confidence Twi prediction ({confidence:.2f}) — "
            f"falling back to 'english'. Text preview: {cleaned[:60]}..."
        )
        return "english", confidence

    internal_lang = _FASTTEXT_TO_INTERNAL.get(label, "english")
    logger.debug(f"fasttext: '{label}' (conf={confidence:.2f}) → '{internal_lang}'")
    return internal_lang, confidence


def _predict_one(model, cleaned: str) -> tuple[list[str], list[float]]:
    """
    Return fasttext's top prediction, tolerating fasttext-wheel 0.9.2 on
    NumPy 2.x.

    fasttext-wheel's Python wrapper still calls np.array(probs, copy=False),
    which raises under NumPy 2 when a copy is unavoidable. The lower-level
    binding returns the same predictions before that conversion, so use it as
    a narrow compatibility fallback instead of patching site-packages.
    """
    try:
        labels, confidences = model.predict(cleaned, k=1)
        return list(labels), list(confidences)
    except ValueError as e:
        if "Unable to avoid copy while creating an array" not in str(e):
            raise
        if not hasattr(model, "f"):
            raise

        predictions = model.f.predict(cleaned + "\n", 1, 0.0, "strict")
        if not predictions:
            return [], []

        probs, labels = zip(*predictions)
        return list(labels), list(probs)


def is_model_unavailable() -> bool:
    """
    Check (without raising) whether the fasttext model is loadable.
    Useful for tests/CI environments without network access to fetch the
    model file — lets callers skip fasttext-dependent tests cleanly rather
    than failing the whole suite on a missing binary asset.
    """
    try:
        _load_model()
        return False
    except RuntimeError:
        return True
