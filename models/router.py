# models/router.py
# MediSafe-GH Â· G-MASS Project
# Team D â€” Engineering Lead
#
# Unified model router for all 5 evaluation models.
# - LLaMA 3.2 3B  â†’ HuggingFace Inference Router (router.huggingface.co/v1)
# - Phi-3 Mini    â†’ HuggingFace Inference Router (router.huggingface.co/v1)
# - BioMistral    â†’ HuggingFace Inference Router (router.huggingface.co/v1)
# - GPT-4o        â†’ OpenAI API
# - Gemini        â†’ Google GenAI API (new SDK)
#
# Usage:
#   from models.router import call_model
#   response = call_model("llama", "Your prompt here")

import os
import re
import time
from dotenv import load_dotenv

load_dotenv()

# â”€â”€ API credentials â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
PHI3_MODEL = os.getenv("PHI3_MODEL", "microsoft/Phi-3-mini-4k-instruct")
BIOMISTRAL_MODEL = os.getenv("BIOMISTRAL_MODEL", "BioMistral/BioMistral-7B-SLERP")
LOCAL_MODEL_BACKEND = os.getenv("LOCAL_MODEL_BACKEND", "hf_router").lower()
PHI3_BACKEND = os.getenv("PHI3_BACKEND", LOCAL_MODEL_BACKEND).lower()
BIOMISTRAL_BACKEND = os.getenv("BIOMISTRAL_BACKEND", LOCAL_MODEL_BACKEND).lower()
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "512"))
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0"))
LOCAL_DEVICE_MAP = os.getenv("LOCAL_DEVICE_MAP", "auto")
LOCAL_TORCH_DTYPE = os.getenv("LOCAL_TORCH_DTYPE", "auto")
LOCAL_QUANTIZATION = os.getenv("LOCAL_QUANTIZATION", "none").lower()
LOCAL_QUANTIZATION_FALLBACK = os.getenv(
    "LOCAL_QUANTIZATION_FALLBACK",
    "true",
).lower() in ("1", "true", "yes")
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


# â”€â”€ Language-consistency instruction (clarifications Â§8) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    Append the Â§8 language-consistency instruction for non-English probes.
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HUGGINGFACE INFERENCE ROUTER  (LLaMA Â· Phi-3 Â· BioMistral)
# Endpoint: https://router.huggingface.co/v1  (OpenAI-compatible)
# No local downloads â€” models run on HuggingFace servers
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def call_hf_model(model_id: str, prompt: str) -> str:
    """
    Calls HuggingFace's Inference Router using the OpenAI-compatible API.
    No local download needed â€” model runs on HuggingFace servers.

    Args:
        model_id : full HuggingFace model ID e.g. "meta-llama/Llama-3.2-3B-Instruct"
        prompt   : the text prompt to send

    Returns:
        The model's generated text as a string.
    """
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is missing. Add it to your .env file.\n"
            "Get one at: huggingface.co â†’ Settings â†’ Access Tokens"
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
    non_retryable_markers = (
        "model_not_supported",
        "not supported by any provider",
        "invalid_request_error",
        "401",
        "403",
        "unauthorized",
        "forbidden",
    )
    if any(marker in message for marker in non_retryable_markers):
        return False
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# LOCAL OPEN-WEIGHT MODELS  (Phi-3 Â· BioMistral)
# Supports:
# - hf_router      â†’ Hugging Face Inference Router
# - local_openai   â†’ local OpenAI-compatible server such as vLLM
# - transformers   â†’ direct local transformers loading
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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

    model_kwargs = _resolve_local_transformers_model_kwargs(torch)
    cache_key = (
        model_id,
        model_kwargs.get("device_map"),
        model_kwargs.get("dtype"),
        LOCAL_ATTN_IMPLEMENTATION,
        LOCAL_TRUST_REMOTE_CODE,
        LOCAL_QUANTIZATION,
    )
    if cache_key not in _TRANSFORMERS_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=LOCAL_TRUST_REMOTE_CODE,
        )
        model_kwargs["trust_remote_code"] = LOCAL_TRUST_REMOTE_CODE
        if LOCAL_ATTN_IMPLEMENTATION:
            model_kwargs["attn_implementation"] = LOCAL_ATTN_IMPLEMENTATION

        model = _load_transformers_model_with_optional_fallback(
            AutoModelForCausalLM,
            model_id,
            model_kwargs,
        )
        if model_kwargs.get("device_map") is None and torch.cuda.is_available():
            model.to("cuda")
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


def _resolve_local_transformers_model_kwargs(torch) -> dict:
    """
    Resolve safe local model-loading kwargs for open-weight models.

    On GPU machines, allow Accelerate's automatic placement. On CPU-only
    machines, avoid device_map='auto' because it can silently choose disk
    offload, which has caused native Windows crashes during generation.
    """
    device_override = os.getenv("LOCAL_DEVICE_MAP")
    dtype_override = os.getenv("LOCAL_TORCH_DTYPE", "auto")
    kwargs = {}

    if device_override:
        requested_device_map = device_override.lower()
        if requested_device_map in ("none", "cpu"):
            device_map = None
        elif requested_device_map == "auto" and not torch.cuda.is_available():
            device_map = None
        else:
            device_map = device_override
    elif torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = None

    if dtype_override != "auto":
        dtype = _resolve_torch_dtype(torch, dtype_override)
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    if device_map is not None:
        kwargs["device_map"] = device_map
    if dtype is not None:
        kwargs["dtype"] = dtype
    quantization_config = _resolve_transformers_quantization_config()
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def _load_transformers_model_with_optional_fallback(model_cls, model_id: str, model_kwargs: dict):
    """Load via Transformers, retrying unquantized if optional quantization fails."""
    try:
        return model_cls.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        if "quantization_config" not in model_kwargs or not LOCAL_QUANTIZATION_FALLBACK:
            raise

        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs.pop("quantization_config", None)
        print(
            f"  Optional local quantization '{LOCAL_QUANTIZATION}' failed for {model_id}; "
            "falling back to the original Transformers loader."
        )
        print(f"  Quantization failure detail: {str(e)[:180]}")
        return model_cls.from_pretrained(model_id, **fallback_kwargs)


def _resolve_transformers_quantization_config():
    """Return an optional Transformers quantization config, or None."""
    if LOCAL_QUANTIZATION in ("", "none", "false", "0"):
        return None

    if LOCAL_QUANTIZATION.startswith("quanto_"):
        try:
            from transformers import QuantoConfig
        except ImportError as e:
            if LOCAL_QUANTIZATION_FALLBACK:
                print(
                    f"  LOCAL_QUANTIZATION={LOCAL_QUANTIZATION} requested, but QuantoConfig "
                    "is unavailable; using the original Transformers loader."
                )
                return None
            raise EnvironmentError(
                "LOCAL_QUANTIZATION requires a Transformers build with QuantoConfig."
            ) from e

        weights = LOCAL_QUANTIZATION.removeprefix("quanto_")
        return QuantoConfig(weights=weights)

    if LOCAL_QUANTIZATION.startswith("bnb_"):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            if LOCAL_QUANTIZATION_FALLBACK:
                print(
                    f"  LOCAL_QUANTIZATION={LOCAL_QUANTIZATION} requested, but "
                    "BitsAndBytesConfig is unavailable; using the original Transformers loader."
                )
                return None
            raise EnvironmentError(
                "LOCAL_QUANTIZATION=bnb_* requires bitsandbytes-compatible Transformers support."
            ) from e

        mode = LOCAL_QUANTIZATION.removeprefix("bnb_")
        if mode == "4bit":
            return BitsAndBytesConfig(load_in_4bit=True)
        if mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)

    raise ValueError(
        "Unknown LOCAL_QUANTIZATION value. Use none, quanto_int8, quanto_int4, "
        "bnb_8bit, or bnb_4bit."
    )


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


# â”€â”€ Individual HF model wrappers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# OPENAI API  (GPT-4o â€” reinstated per explicit team decision, overriding Â§9)
#
# Â§9 of GMASS_Team_Clarifications.md recommended GPT-4o mini (94% cheaper,
# comparable safety-classification performance, ~$5 total for all 1,800
# proprietary calls). The team explicitly chose to reinstate full GPT-4o
# instead, to match the original 5-model lineup. Cost impact: full GPT-4o is
# significantly more per-token than GPT-4o mini â€” budget accordingly for the
# 900 GPT-4o calls in a full run; confirm against current OpenAI pricing
# before a production run, as mini's <$5 estimate no longer applies.
#
# To switch back to mini without code changes, set GPT4O_MODEL=gpt-4o-mini
# in .env â€” the model_id is fully configurable, only the default changed.
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

GPT4O_MODEL = os.getenv("GPT4O_MODEL", "gpt-4o")


def call_gpt4o(prompt: str) -> str:
    """
    GPT-4o via OpenAI API. Reinstated per explicit team decision (see module
    comment above) â€” overrides clarifications Â§9's GPT-4o mini recommendation.
    Requires OPENAI_API_KEY in .env.
    Get key at: platform.openai.com/api-keys

    NOTE: function name kept as call_gpt4o / model key kept as "gpt4o" for
    backward compatibility with existing pipeline code, configs, and scored
    output files. The MODEL_ID actually used is controlled by GPT4O_MODEL â€”
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# GOOGLE API  (Gemini)
# Uses new google-genai SDK (google-generativeai is deprecated)
# Get key at: aistudio.google.com
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def call_gemini(prompt: str) -> str:
    """
    Gemini via Google GenAI API (new SDK).
    Requires GEMINI_API_KEY in .env.
    Free tier: 1,500 requests/day.
    Get key at: aistudio.google.com â†’ Get API Key
    """
    if not GEMINI_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to your .env file.\n"
            "Get one at: aistudio.google.com â†’ Get API Key"
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


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# UNIFIED DISPATCHER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

MODEL_FUNCTIONS = {
    "gpt4o":      call_gpt4o,
    "gemini":     call_gemini,
    "phi3":       call_phi3,
    "biomistral": call_biomistral,
}

VALID_MODELS = list(MODEL_FUNCTIONS.keys())


def call_model(model_name: str, prompt: str) -> str:
    """
    Universal entry point. Use this from your scoring pipeline.

    Args:
        model_name : one of "gpt4o", "gemini", "phi3", "biomistral"
        prompt     : the text prompt to send

    Returns:
        The model's response as a plain string.

    Example:
        from models.router import call_model
        response = call_model("gemini", "What are symptoms of malaria?")
    """
    model_name = normalize_model_name(model_name)
    fn = MODEL_FUNCTIONS.get(model_name)
    if fn is None:
        raise ValueError(
            f"Unknown model: '{model_name}'.\n"
            f"Valid options: {VALID_MODELS}"
        )
    return fn(prompt)



