# G-MASS OpenRouter Unified API Integration Plan
**MediSafe-GH · Track II Africa AI Safety Prize · KNUST Bioinstrumentation & Medical Imaging Laboratory**

---

## 1. Overview & Objectives

Currently, evaluating the full suite of target models in G-MASS requires researchers to provision and configure separate API credentials across multiple providers:
- `OPENAI_API_KEY` (for GPT-4o / GPT-4o Mini)
- `GEMINI_API_KEY` (for Gemini 2.5 Flash & Hosted Policy Judge)
- `HF_TOKEN` (for Hugging Face hosted inference endpoints)
- `KHAYA_API_KEY` (for GhanaNLP neural translation)

Adopting **OpenRouter** (`https://openrouter.ai/`) as an optional unified gateway allows researchers and clinical testers to run evaluations across all frontier and open-weight models using a **single unified API key** (`OPENROUTER_API_KEY`), drastically lowering adoption barriers.

---

## 2. Model Mapping Matrix via OpenRouter

OpenRouter exposes OpenAI-compatible REST endpoints (`https://openrouter.ai/api/v1/chat/completions`), enabling a single adapter in `models/router.py`:

| G-MASS Model Key | Target Architecture | OpenRouter Model Slug | Direct API Equivalent | Status on OpenRouter |
|---|---|---|---|:---:|
| `gpt4o` | GPT-4o Mini | `openai/gpt-4o-mini` | OpenAI API | ✅ Fully Supported |
| `gemini` | Gemini 2.5 Flash | `google/gemini-2.5-flash` | Google AI Studio | ✅ Fully Supported |
| `phi3` | Phi-3 Mini 4K | `microsoft/phi-3-mini-128k-instruct` | Local / HF Endpoint | ✅ Fully Supported |
| `biomistral` | BioMistral 7B / Medical Mistral | `mistralai/mistral-7b-instruct` | Local Transformers | ✅ Supported (or Hugging Face fallback) |

---

## 3. Back-Translation Architecture: Khaya vs. OpenRouter

A crucial architectural consideration raised in the design is whether OpenRouter can also replace `KHAYA_API_KEY` for Akan/Twi back-translation.

### Clarification on Khaya & GhanaNLP:
- **Khaya** is GhanaNLP's proprietary transformer-based translation service specifically trained on high-volume Akan, Ga, Ewe, and Dagbani corpora.
- **Khaya is not hosted on OpenRouter** (OpenRouter only hosts general-purpose LLMs).

### Recommended Dual-Engine Translation Strategy:

```
                  [ Twi Response Received ]
                              │
               Is KHAYA_API_KEY configured?
                     /                  \
                  YES                    NO
                  /                        \
       [ Tier 1: Khaya API ]     Is OPENROUTER_API_KEY configured?
       (Gold Standard NMT)                 /             \
                                        YES               NO
                                        /                   \
                        [ Tier 2: OpenRouter LLM ]    [ Tier 3: Error / Fallback ]
                        - google/gemini-2.5-flash     (AfroLM-only native scoring)
                        - meta-llama/llama-3.1-70b
                        (Zero-shot translation)
```

1. **Tier 1 (Gold Standard)**: If `KHAYA_API_KEY` is present, the pipeline invokes GhanaNLP's dedicated translation model.
2. **Tier 2 (Unified OpenRouter Fallback)**: If only `OPENROUTER_API_KEY` is provided, G-MASS automatically uses a high-capability multilingual model on OpenRouter (`google/gemini-2.5-flash` or `meta-llama/llama-3.1-70b-instruct`) with a standardized medical translation system prompt:
   ```text
   You are an expert bilingual medical translator in Ghana.
   Translate the following Akan (Twi) clinical text faithfully into English.
   Preserve all medical instructions, drug dosages, and clinical referral advice exactly.
   Do not add commentary.
   ```
3. **Tier 3 (Zero Key / Offline)**: If neither is configured, the pipeline scores Twi natively using AfroLM without requiring English translation.

---

## 4. Proposed Implementation Architecture (`models/router.py`)

In `models/router.py`, introduce an `OpenRouterCaller`:

```python
class OpenRouterCaller:
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
    def call(self, model_slug: str, prompt: str, temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/Armstrong66/medisafe-gh",
            "X-Title": "G-MASS Medical AI Safety Screen",
        }
        # Standard OpenAI client with custom base_url
```

In `configs/gmass_config.yaml`:
```yaml
api_gateways:
  default_provider: direct  # Options: direct | openrouter
  openrouter:
    enabled: true
    models:
      gpt4o: openai/gpt-4o-mini
      gemini: google/gemini-2.5-flash
      phi3: microsoft/phi-3-mini-128k-instruct
```

In Gradio UI Settings Tab:
- Add a single `OpenRouter API Key (Unified)` input box that auto-activates all model options when filled.

---

## 5. Benefits Summary
- **Simplicity**: Users need only one account and one credit balance.
- **Fairness**: Identical network latency and sampling parameters across all evaluated models.
- **Graceful Degradation**: Dual-track translation ensures research progress even without separate GhanaNLP accounts.
