# MediSafe-GH · G-MASS

**Ghana Medical AI Safety Screen** — an open-source evaluation protocol that
tests whether AI health assistants behave safely when queried in Twi and
Ghanaian English.

Africa AI Safety Prize Competition 2026 · Track II submission.

---

## Phase 0 Quick Start (Windows)

### 1. Clone and set up environment

```powershell
git clone https://github.com/YOUR-USERNAME/medisafe-gh.git
cd medisafe-gh

python -m venv medisafe-env
medisafe-env\Scripts\activate

pip install -r requirements.txt
```

### 2. Create your .env file

```powershell
copy .env.example .env
```

Open `.env` in Notepad and fill in your three API keys:

| Key | Where to get it |
|-----|----------------|
| `HF_TOKEN` | huggingface.co → Settings → Access Tokens |
| `OPENAI_API_KEY` | platform.openai.com/api-keys |
| `GEMINI_API_KEY` | aistudio.google.com → Get API Key |

### 3. Check your environment

```powershell
python setup.py
```

### 4. Test all 5 models

```powershell
python test_models.py
```

Test a single model:

```powershell
python test_models.py llama
python test_models.py phi3
python test_models.py biomistral
python test_models.py gpt4o
python test_models.py gemini
```

---

## Running Phi-3 and BioMistral Locally

Phi-3 and BioMistral are open-weight models. They can use the Hugging Face
router, a local OpenAI-compatible server, or direct local `transformers`
loading.

### Option A: direct local Transformers

Install the optional local dependencies:

```powershell
pip install -r requirements-local.txt
```

Add this to `.env`:

```env
PHI3_BACKEND=transformers
BIOMISTRAL_BACKEND=transformers
LOCAL_DEVICE_MAP=auto
LOCAL_TORCH_DTYPE=auto
LOCAL_MAX_NEW_TOKENS=512
LOCAL_TEMPERATURE=0
```

Then run:

```powershell
python test_models.py phi3
python test_models.py biomistral
```

BioMistral is a 7B model, so it may need a GPU or a quantized local runtime.

### Option B: local OpenAI-compatible server

Use this if you run the models with a server such as vLLM. Add this to `.env`:

```env
PHI3_BACKEND=local_openai
PHI3_LOCAL_BASE_URL=http://localhost:8000/v1
PHI3_LOCAL_MODEL=microsoft/Phi-3-mini-4k-instruct

BIOMISTRAL_BACKEND=local_openai
BIOMISTRAL_LOCAL_BASE_URL=http://localhost:8001/v1
BIOMISTRAL_LOCAL_MODEL=BioMistral/BioMistral-7B-SLERP
```

The public router names stay the same:

```powershell
python test_models.py phi3
python test_models.py biomistral
```

---

## Models

| Model | Provider | Via |
|-------|----------|-----|
| LLaMA 3.2 3B Instruct | Meta | HuggingFace Inference API |
| Phi-3 Mini 4K Instruct | Microsoft | HuggingFace Inference API |
| BioMistral 7B SLERP | BioMistral | HuggingFace Inference API |
| GPT-4o | OpenAI | OpenAI API |
| Gemini 1.5 Flash | Google | Google Generative AI API |

---

## Using the Router in Your Code

```python
from models.router import call_model

response = call_model("llama", "What are the symptoms of malaria?")
print(response)
```

Valid model names: `llama`, `phi3`, `biomistral`, `gpt4o`, `gemini`

---

## Licence

Apache 2.0
