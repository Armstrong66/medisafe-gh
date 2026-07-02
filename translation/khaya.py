"""
translation.khaya — Khaya / GhanaNLP translation bridge.
Owner: D (Engineering Lead) + C (Translation Lead)
MediSafe-GH · Africa AI Safety Prize 2026

REVISED per GMASS_Team_Clarifications.md §7 — this module's role narrowed.
It previously fed BOTH LlamaGuard3 and RoBERTa. RoBERTa has been replaced
by AfroLM (a Twi-trained multilingual model — see scorer.py), which scores
Twi responses NATIVELY and needs no translation at all. Khaya now feeds
ONLY LlamaGuard3, which remains English-only and acts as the SECONDARY
cross-validator for Twi (AfroLM is primary for Twi).

WHY THIS MODULE STILL EXISTS
------------------------------
LlamaGuard3 was trained on English safety taxonomy (MLCommons S1–S14) and
needs BOTH the probe and the model's response in English to classify
correctly. AfroLM, by contrast, is natively Twi-trained and scores the
original Twi response directly — it never calls into this module.

So when a model answers a Twi probe in Twi, Khaya translates that response
to English ONLY for LlamaGuard3's benefit:

    Twi probe ──► Model ──► Twi response
                                  │
                    ┌─────────────┴─────────────┐
                    │                            │
                    ▼                            ▼
            AfroLMScorer                 [ KHAYA TRANSLATOR ]
        (scores ORIGINAL Twi                      │
         directly — PRIMARY                       ▼
         for Twi, no Khaya call)           English response
                                                   │
                                                   ▼
                                          LlamaGuard3Scorer
                                       (probe_en + response_en)
                                       SECONDARY for Twi, via
                                       this back-translation

ReferralDetector and HallucinationDetector in scorer.py do NOT go through
Khaya — they run on the ORIGINAL Twi response, because their keyword
lists include Twi referral phrases (e.g. "kɔ dokita" = "go to doctor").
Translating first would risk losing those exact phrasings.

FALLBACK CHAIN
---------------
1. Khaya API (hosted, requires KHAYA_API_KEY)  — primary
2. Local NLLB model (no key needed)            — fallback
3. Raise on total failure — never silently score Twi text as if English

Usage:
    from translation.khaya import translate_response_to_english

    response_en = translate_response_to_english(
        text     = model_response_twi,
        language = "twi",
    )
    # response_en is now safe to pass into LlamaGuard3Scorer
"""

import json
import os
from pathlib import Path
from core.logger import get_logger
from core.utils import retry_with_backoff

logger = get_logger(__name__)

KHAYA_API_URL  = "https://translation-api.ghananlp.org/v1/translate"
NLLB_MODEL = os.getenv("NLLB_TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_SOURCE_LANG = os.getenv("NLLB_SOURCE_LANG_TWI", "aka_Latn")
NLLB_TARGET_LANG = os.getenv("NLLB_TARGET_LANG_EN", "eng_Latn")
NLLB_MAX_NEW_TOKENS = int(os.getenv("NLLB_MAX_NEW_TOKENS", "256"))
# Languages that require translation before scoring.
# "english" never needs translation — pass through.
TRANSLATION_REQUIRED_LANGUAGES = {"twi"}

# Khaya / GhanaNLP language codes
_KHAYA_LANG_CODES = {
    "twi": "tw",
    "ga":  "gaa",
    "ewe": "ee",
    "fante": "fat",
    "dagbani": "dag",
}

# Cache: avoid re-translating identical text within one run
_translation_cache: dict[str, str] = {}


def _load_khaya_api_key() -> str | None:
    """Load Khaya key from env or a local GhanaNLP credential JSON file."""
    direct_key = os.getenv("KHAYA_API_KEY")
    if direct_key:
        return direct_key

    credentials_path = os.getenv("KHAYA_CREDENTIALS_PATH") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        return None

    try:
        with open(Path(credentials_path), encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read KHAYA_CREDENTIALS_PATH: {e}")
        return None

    for key_name in ("KHAYA_API_KEY", "khaya_api_key", "api_key", "subscription_key", "Ocp-Apim-Subscription-Key"):
        value = payload.get(key_name)
        if value:
            return str(value)

    logger.warning("Khaya credential JSON did not contain a recognized API key field")
    return None


def _extract_translation(payload: dict) -> str:
    """Handle the response shapes returned by GhanaNLP/Khaya deployments."""
    for key in ("translatedText", "translation", "translated_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_translation(data)

    translations = payload.get("translations")
    if isinstance(translations, list) and translations:
        first = translations[0]
        if isinstance(first, dict):
            return _extract_translation(first)
        if isinstance(first, str):
            return first.strip()

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY — Khaya hosted API (twi → english)
# ══════════════════════════════════════════════════════════════════════════════

def _translate_via_khaya_api(text: str, source_lang: str = "tw") -> str | None:
    """
    Translate text to English via the hosted Khaya API.

    Args:
        text        : source text (Twi, Ga, Ewe, etc.)
        source_lang : Khaya language code (default "tw" = Twi)

    Returns:
        English translation, or None on failure (triggers fallback).
    """
    api_key = _load_khaya_api_key()
    if not api_key:
        logger.debug("KHAYA_API_KEY/KHAYA_CREDENTIALS_PATH not set; skipping hosted API")
        return None

    import requests

    def _call():
        response = requests.post(
            KHAYA_API_URL,
            json={"text": text, "in": source_lang, "out": "en"},
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": api_key,
            },
            timeout=20,
        )
        response.raise_for_status()
        return _extract_translation(response.json())

    result = retry_with_backoff(_call, retries=2, base_wait=2.0)
    if result:
        logger.debug(f"Khaya API translation succeeded ({len(result)} chars)")
    return result or None


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK — local NLLB model (aka_Latn → eng_Latn)
# ══════════════════════════════════════════════════════════════════════════════

_local_translator = None


def _translate_via_local_model(text: str) -> str | None:
    """
    Fallback: translate Twi/Akan → English using a local NLLB model.
    Runs locally, no API key needed.

    Default model: facebook/nllb-200-distilled-600M
    """
    global _local_translator
    try:
        if _local_translator is None:
            from transformers import pipeline as hf_pipeline
            from core.utils import get_device
            device = 0 if get_device() == "cuda" else -1
            _local_translator = hf_pipeline(
                "translation",
                model=NLLB_MODEL,
                device=device,
                local_files_only=True,
            )
            logger.info(f"NLLB local Twi→English model loaded from {NLLB_MODEL}")

        result = _local_translator(
            text,
            src_lang=NLLB_SOURCE_LANG,
            tgt_lang=NLLB_TARGET_LANG,
            max_new_tokens=NLLB_MAX_NEW_TOKENS,
        )[0]["translation_text"]
        logger.debug(f"Local NLLB translation succeeded ({len(result)} chars)")
        return result

    except Exception as e:
        logger.error(
            f"Local NLLB translation failed for model '{NLLB_MODEL}': {e}. "
            "The fallback runs in offline mode; pre-download the model into the Hugging Face "
            "cache or set NLLB_TRANSLATION_MODEL to a local model directory."
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def translate_response_to_english(text: str, language: str) -> str:
    """
    Translate a model's response to English before it reaches the safety
    scorer (LlamaGuard3). This is the function scorer.py calls for detected
    Twi responses before the LlamaGuard3 cross-check.

    If language == "english", returns text unchanged (no-op, no API call).

    Fallback chain: Khaya hosted API → local NLLB model → raise.
    Results are cached per-process to avoid re-translating identical text.

    Args:
        text     : the model's response in its original language
        language : "english", "twi", or "ghanaian_en"

    Returns:
        English text, safe to pass to LlamaGuard3Scorer.score().

    Raises:
        RuntimeError if both Khaya and the local fallback fail — we must
        NEVER silently score un-translated Twi text as if it were English.

    Example:
        response_en = translate_response_to_english(twi_response, "twi")
        lg3_result  = lg3_scorer.score(probe_id, model_id, "twi", probe_en, response_en)
    """
    if language not in TRANSLATION_REQUIRED_LANGUAGES:
        return text  # english — no translation needed

    cache_key = f"{language}:{text}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    source_lang = _KHAYA_LANG_CODES.get(language, "tw")

    # 1) Try Khaya hosted API
    translated = _translate_via_khaya_api(text, source_lang)

    # 2) Fall back to local NLLB model
    if translated is None:
        logger.info("Falling back to local NLLB translation model")
        translated = _translate_via_local_model(text)

    # 3) Total failure — do not let un-translated Twi reach the scorers
    if translated is None:
        raise RuntimeError(
            f"Translation failed for language='{language}' via both Khaya API "
            f"and local fallback. Refusing to pass untranslated Twi text to "
            f"LlamaGuard3 — it is not trained on Twi and would "
            f"silently misclassify it. Text preview: {text[:80]}..."
        )

    _translation_cache[cache_key] = translated
    return translated


def clear_translation_cache() -> None:
    """Clear the in-memory translation cache. Useful between test runs."""
    global _translation_cache
    _translation_cache = {}
    logger.debug("Translation cache cleared")
