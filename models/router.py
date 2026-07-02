# models/router.py
# MediSafe-GH · G-MASS Project
# Team D — Engineering Lead
#
# Unified model router for all 5 evaluation models.
# - LLaMA 3.2 3B  → HuggingFace Inference Router (router.huggingface.co/v1)
# - Phi-3 Mini    → HuggingFace Inference Router (router.huggingface.co/v1)
# - BioMistral    → HuggingFace Inference Router (router.huggingface.co/v1)
# - GPT-4o        → OpenAI API
# - Gemini        → Google GenAI API (new SDK)
#
# Usage:
#   from models.router import call_model
#   response = call_model("llama", "Your prompt here")

import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# ── API credentials ────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-2.0-flash",
    ).split(",")
    if model.strip()
]
GEMINI_RETRIES = int(os.getenv("GEMINI_RETRIES", "4"))
GEMINI_RETRY_DELAY = float(os.getenv("GEMINI_RETRY_DELAY", "2"))
HF_RETRIES = int(os.getenv("HF_RETRIES", "4"))
HF_RETRY_DELAY = float(os.getenv("HF_RETRY_DELAY", "2"))
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
LLAMA_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "LLAMA_FALLBACK_MODELS",
        "meta-llama/Llama-3.1-8B-Instruct",
    ).split(",")
    if model.strip()
]
PHI3_MODEL = os.getenv("PHI3_MODEL", "microsoft/Phi-3-mini-4k-instruct")
BIOMISTRAL_MODEL = os.getenv("BIOMISTRAL_MODEL", "BioMistral/BioMistral-7B-SLERP")
LOCAL_MODEL_BACKEND = os.getenv("LOCAL_MODEL_BACKEND", "hf_router").lower()
PHI3_BACKEND = os.getenv("PHI3_BACKEND", LOCAL_MODEL_BACKEND).lower()
BIOMISTRAL_BACKEND = os.getenv("BIOMISTRAL_BACKEND", LOCAL_MODEL_BACKEND).lower()
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "512"))
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0"))
LOCAL_DEVICE_MAP = os.getenv("LOCAL_DEVICE_MAP", "auto")
LOCAL_TORCH_DTYPE = os.getenv("LOCAL_TORCH_DTYPE", "auto")
LOCAL_ATTN_IMPLEMENTATION = os.getenv("LOCAL_ATTN_IMPLEMENTATION", "eager")
LOCAL_TRUST_REMOTE_CODE = os.getenv("LOCAL_TRUST_REMOTE_CODE", "false").lower() in (
    "1",
    "true",
    "yes",
)
PHI3_LOCAL_BASE_URL = os.getenv("PHI3_LOCAL_BASE_URL", "http://localhost:8000/v1")
BIOMISTRAL_LOCAL_BASE_URL = os.getenv("BIOMISTRAL_LOCAL_BASE_URL", "http://localhost:8001/v1")
PHI3_LOCAL_MODEL = os.getenv("PHI3_LOCAL_MODEL", PHI3_MODEL)
BIOMISTRAL_LOCAL_MODEL = os.getenv("BIOMISTRAL_LOCAL_MODEL", BIOMISTRAL_MODEL)

_TRANSFORMERS_CACHE = {}


# ── Language-consistency instruction (clarifications §8) ──────────────────────
# Frontier models often default to English even when prompted in Twi.
# Appending this instruction forces language-consistent responses where the
# model is capable of complying, and surfaces non-compliance as a documented
# finding where it is not (checked downstream by LanguageConsistencyChecker
# in scorer/scorer.py).
_DEFAULT_LANGUAGE_CONSISTENCY_INSTRUCTION = (
    "\n\nPlease respond in the same language as the question."
)

_LANGUAGE_SPECIFIC_INSTRUCTIONS = {
    "twi": (
        "\n\nRespond only in Twi/Akan. Do not answer in English, and do not "
        "translate the question into English."
    ),
    "ghanaian_en": (
        "\n\nRespond in Ghanaian English. Keep the answer in English, using "
        "natural Ghanaian wording where appropriate."
    ),
}

# Languages that require the consistency instruction. English needs no
# instruction since it's almost always the model's default behaviour anyway.
_LANGUAGES_REQUIRING_INSTRUCTION = {"twi", "ghanaian_en"}


def build_prompt_with_language_instruction(prompt: str, language: str = "english") -> str:
    """
    Append the §8 language-consistency instruction for non-English probes.
    No-op for English. Call this BEFORE passing a prompt to call_model()
    when evaluating Twi or Ghanaian English conditions.

    Args:
        prompt   : the original probe text
        language : "english", "twi", or "ghanaian_en"

    Returns:
        The prompt, with the consistency instruction appended if needed.

    Example:
        prompt = build_prompt_with_language_instruction(twi_prompt, "twi")
        response = call_model("gemini", prompt)
    """
    if language in _LANGUAGES_REQUIRING_INSTRUCTION:
        return prompt + _LANGUAGE_SPECIFIC_INSTRUCTIONS.get(
            language,
            _DEFAULT_LANGUAGE_CONSISTENCY_INSTRUCTION,
        )
    return prompt


def normalize_model_name(model_name: str) -> str:
    """Return the canonical model key used by the router."""
    return str(model_name).strip().lower()


def clean_model_response(text: str) -> str:
    """Remove common chat-template artifacts from model outputs."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return cleaned

    cleaned = re.sub(r"<\|/?(?:assistant|user|system)\|>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"<\|(?:end|eot|endoftext)\|>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(?:assistant|ai|model)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.split(r"\n\s*(?:User|Patient)\s*:", cleaned, maxsplit=1)[0]
    return cleaned.strip()


# ══════════════════════════════════════════════════════════════════════════════
# HUGGINGFACE INFERENCE ROUTER  (LLaMA · Phi-3 · BioMistral)
# Endpoint: https://router.huggingface.co/v1  (OpenAI-compatible)
# No local downloads — models run on HuggingFace servers
# ══════════════════════════════════════════════════════════════════════════════

def call_hf_model(model_id: str, prompt: str) -> str:
    """
    Calls HuggingFace's Inference Router using the OpenAI-compatible API.
    No local download needed — model runs on HuggingFace servers.

    Args:
        model_id : full HuggingFace model ID e.g. "meta-llama/Llama-3.2-3B-Instruct"
        prompt   : the text prompt to send

    Returns:
        The model's generated text as a string.
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is missing. Add it to your .env file.\n"
            "Get one at: huggingface.co → Settings → Access Tokens"
        )

    from openai import OpenAI

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

    last_error = None
    for attempt in range(1, HF_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            text = clean_model_response(response.choices[0].message.content)
            if not text:
                raise RuntimeError(f"{model_id} returned an empty response.")
            return text
        except Exception as e:
            last_error = e
            if not _is_retryable_hf_error(e) or attempt == HF_RETRIES:
                break

            delay = HF_RETRY_DELAY * (2 ** (attempt - 1))
            print(
                f"  HuggingFace transient error on {model_id}; "
                f"retrying in {delay:.1f}s ({attempt}/{HF_RETRIES})..."
            )
            time.sleep(delay)

    raise last_error


def _is_retryable_hf_error(error: Exception) -> bool:
    """Return True for temporary Hugging Face router/provider failures."""
    message = str(error).lower()
    retryable_markers = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "model is loading",
        "provider",
        "overloaded",
    )
    return any(marker in message for marker in retryable_markers)


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL OPEN-WEIGHT MODELS  (Phi-3 · BioMistral)
# Supports:
# - hf_router      → Hugging Face Inference Router
# - local_openai   → local OpenAI-compatible server such as vLLM
# - transformers   → direct local transformers loading
# ══════════════════════════════════════════════════════════════════════════════

def call_open_weight_model(
    backend: str,
    model_id: str,
    prompt: str,
    local_base_url: str,
    local_model_id: str,
) -> str:
    if backend == "hf_router":
        return call_hf_model(model_id, prompt)
    if backend == "local_openai":
        return call_local_openai_model(local_base_url, local_model_id, prompt)
    if backend == "transformers":
        return call_transformers_model(model_id, prompt)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Use one of: hf_router, local_openai, transformers."
    )


def call_local_openai_model(base_url: str, model_id: str, prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("LOCAL_OPENAI_API_KEY", "local"),
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=LOCAL_MAX_NEW_TOKENS,
        temperature=LOCAL_TEMPERATURE,
    )
    text = clean_model_response(response.choices[0].message.content)
    if not text:
        raise RuntimeError(f"{model_id} returned an empty response from {base_url}.")
    return text


def call_transformers_model(model_id: str, prompt: str) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise EnvironmentError(
            "Local transformers backend requires torch, transformers, and accelerate.\n"
            "Install with: pip install -r requirements-local.txt"
        ) from e

    cache_key = (model_id, LOCAL_DEVICE_MAP, LOCAL_TORCH_DTYPE)
    if cache_key not in _TRANSFORMERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=LOCAL_TRUST_REMOTE_CODE,
        )
        model_kwargs = {
            "device_map": LOCAL_DEVICE_MAP,
            "trust_remote_code": LOCAL_TRUST_REMOTE_CODE,
        }
        if LOCAL_ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = LOCAL_ATTN_IMPLEMENTATION
        torch_dtype = _resolve_torch_dtype(torch, LOCAL_TORCH_DTYPE)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        _TRANSFORMERS_CACHE[cache_key] = (tokenizer, model)

    tokenizer, model = _TRANSFORMERS_CACHE[cache_key]
    inputs = _build_transformers_inputs(tokenizer, prompt)
    inputs = _move_inputs_for_generation(model, inputs)

    generation_kwargs = {
        "max_new_tokens": LOCAL_MAX_NEW_TOKENS,
        "do_sample": LOCAL_TEMPERATURE > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if LOCAL_TEMPERATURE > 0:
        generation_kwargs["temperature"] = LOCAL_TEMPERATURE

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    text = clean_model_response(tokenizer.decode(generated_ids, skip_special_tokens=True))
    if not text:
        raise RuntimeError(f"{model_id} returned an empty local response.")
    return text


def _build_transformers_inputs(tokenizer, prompt: str) -> dict:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
    return tokenizer(f"User: {prompt}\nAssistant:", return_tensors="pt")


def _move_inputs_for_generation(model, inputs: dict) -> dict:
    device = None
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        device = next(
            (
                mapped_device
                for mapped_device in device_map.values()
                if mapped_device not in ("cpu", "disk")
            ),
            None,
        )

    if device is None:
        device = getattr(model, "device", None)

    if device is None or str(device) == "disk":
        return inputs

    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _resolve_torch_dtype(torch, dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name in ("none", ""):
        return None
    if hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(
        f"Unknown LOCAL_TORCH_DTYPE '{dtype_name}'. "
        "Common values: auto, float16, bfloat16, float32."
    )


# ── Individual HF model wrappers ──────────────────────────────────────────────

def call_llama(prompt: str) -> str:
    """
    LLaMA 3.2 3B Instruct via HuggingFace Inference Router.
    Reinstated as the original-lineup model (overriding an earlier
    pipeline-availability substitution to Llama-3.1-8B-Instruct).

    Falls back to LLAMA_FALLBACK_MODELS if the primary model is rejected
    outright by the router (e.g. "not supported by any provider you have
    enabled" — a non-retryable error distinct from transient rate limits,
    which call_hf_model's internal retry loop already handles). This
    mirrors call_gemini's fallback pattern and exists specifically because
    3.2-3B has previously been unavailable on the router at times.
    """
    models_to_try = [LLAMA_MODEL] + [
        model for model in LLAMA_FALLBACK_MODELS if model != LLAMA_MODEL
    ]
    last_error = None

    for model_id in models_to_try:
        try:
            return call_hf_model(model_id, prompt)
        except Exception as e:
            last_error = e
            if "not supported by any provider" in str(e).lower() or "model_not_supported" in str(e).lower():
                print(f"  {model_id} unavailable on HF router; trying fallback...")
                continue
            raise  # any other error (auth, retryable-exhausted, etc.) propagates immediately

    raise last_error


def call_phi3(prompt: str) -> str:
    """Phi-3 Mini 4K Instruct via the configured open-weight backend."""
    return call_open_weight_model(
        PHI3_BACKEND,
        PHI3_MODEL,
        prompt,
        PHI3_LOCAL_BASE_URL,
        PHI3_LOCAL_MODEL,
    )


def call_biomistral(prompt: str) -> str:
    """BioMistral 7B SLERP via the configured open-weight backend."""
    return call_open_weight_model(
        BIOMISTRAL_BACKEND,
        BIOMISTRAL_MODEL,
        prompt,
        BIOMISTRAL_LOCAL_BASE_URL,
        BIOMISTRAL_LOCAL_MODEL,
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPENAI API  (GPT-4o — reinstated per explicit team decision, overriding §9)
#
# §9 of GMASS_Team_Clarifications.md recommended GPT-4o mini (94% cheaper,
# comparable safety-classification performance, ~$5 total for all 1,800
# proprietary calls). The team explicitly chose to reinstate full GPT-4o
# instead, to match the original 5-model lineup. Cost impact: full GPT-4o is
# significantly more per-token than GPT-4o mini — budget accordingly for the
# 900 GPT-4o calls in a full run; confirm against current OpenAI pricing
# before a production run, as mini's <$5 estimate no longer applies.
#
# To switch back to mini without code changes, set GPT4O_MODEL=gpt-4o-mini
# in .env — the model_id is fully configurable, only the default changed.
# ══════════════════════════════════════════════════════════════════════════════

GPT4O_MODEL = os.getenv("GPT4O_MODEL", "gpt-4o")


def call_gpt4o(prompt: str) -> str:
    """
    GPT-4o via OpenAI API. Reinstated per explicit team decision (see module
    comment above) — overrides clarifications §9's GPT-4o mini recommendation.
    Requires OPENAI_API_KEY in .env.
    Get key at: platform.openai.com/api-keys

    NOTE: function name kept as call_gpt4o / model key kept as "gpt4o" for
    backward compatibility with existing pipeline code, configs, and scored
    output files. The MODEL_ID actually used is controlled by GPT4O_MODEL —
    see constant above and configs/models.yaml.
    """
    if not OPENAI_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file.\n"
            "Get one at: platform.openai.com/api-keys"
        )

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=GPT4O_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    return clean_model_response(response.choices[0].message.content)


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE API  (Gemini)
# Uses new google-genai SDK (google-generativeai is deprecated)
# Get key at: aistudio.google.com
# ══════════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str) -> str:
    """
    Gemini via Google GenAI API (new SDK).
    Requires GEMINI_API_KEY in .env.
    Free tier: 1,500 requests/day.
    Get key at: aistudio.google.com → Get API Key
    """
    if not GEMINI_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to your .env file.\n"
            "Get one at: aistudio.google.com → Get API Key"
        )

    from google import genai

    client = genai.Client(api_key=GEMINI_KEY)
    models_to_try = [GEMINI_MODEL] + [
        model for model in GEMINI_FALLBACK_MODELS if model != GEMINI_MODEL
    ]
    last_error = None

    for model in models_to_try:
        exhausted_retryable_error = False
        for attempt in range(1, GEMINI_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError(f"{model} returned an empty response.")
                return text
            except Exception as e:
                last_error = e
                if not _is_retryable_gemini_error(e):
                    raise
                if attempt == GEMINI_RETRIES:
                    exhausted_retryable_error = True
                    break

                delay = GEMINI_RETRY_DELAY * (2 ** (attempt - 1))
                print(
                    f"  Gemini transient error on {model}; "
                    f"retrying in {delay:.1f}s ({attempt}/{GEMINI_RETRIES})..."
                )
                time.sleep(delay)

        if exhausted_retryable_error and model != models_to_try[-1]:
            next_model = models_to_try[models_to_try.index(model) + 1]
            print(f"  Gemini fallback: trying {next_model}...")

    raise last_error


def _is_retryable_gemini_error(error: Exception) -> bool:
    """Return True for temporary Gemini API failures worth retrying."""
    message = str(error).lower()
    if _is_non_retryable_gemini_quota_error(error):
        return False
    retryable_markers = (
        "503",
        "unavailable",
        "overloaded",
        "high demand",
        "500",
        "internal",
        "504",
        "deadline_exceeded",
        "429",
        "resource_exhausted",
    )
    return any(marker in message for marker in retryable_markers)


def _is_non_retryable_gemini_quota_error(error: Exception) -> bool:
    """
    Return True for hard quota failures that retries/fallbacks cannot fix.

    Gemini also reports short rate limits as 429 RESOURCE_EXHAUSTED, and those
    are worth retrying. The free-tier "limit: 0" / daily quota messages from
    the API are different: every retry just waits and then fails again.
    """
    message = str(error).lower()
    hard_quota_markers = (
        "free_tier_requests, limit: 0",
        "free_tier_input_token_count, limit: 0",
        "generate requests per day",
        "generate_content_free_tier_requests",
        "check your plan and billing details",
    )
    return "429" in message and any(marker in message for marker in hard_quota_markers)


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

MODEL_FUNCTIONS = {
    "gpt4o":      call_gpt4o,
    "gemini":     call_gemini,
    "llama":      call_llama,
    "phi3":       call_phi3,
    "biomistral": call_biomistral,
}

VALID_MODELS = list(MODEL_FUNCTIONS.keys())


def call_model(model_name: str, prompt: str) -> str:
    """
    Universal entry point. Use this from your scoring pipeline.

    Args:
        model_name : one of "gpt4o", "gemini", "llama", "phi3", "biomistral"
        prompt     : the text prompt to send

    Returns:
        The model's response as a plain string.

    Example:
        from models.router import call_model
        response = call_model("llama", "What are symptoms of malaria?")
    """
    model_name = normalize_model_name(model_name)
    fn = MODEL_FUNCTIONS.get(model_name)
    if fn is None:
        raise ValueError(
            f"Unknown model: '{model_name}'.\n"
            f"Valid options: {VALID_MODELS}"
        )
    return fn(prompt)
