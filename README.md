# MediSafe-GH - G-MASS

**Ghana Medical AI Safety Screen** -- an open-source evaluation pipeline for
testing whether medical AI assistants behave safely when queried in English,
Twi, and Ghanaian English.

Africa AI Safety Prize Competition 2026 - Track II submission.

## What The Pipeline Does

G-MASS evaluates model responses with clinical safety metrics, not just general
helpfulness or factuality.

- **CSR**: Clinical Safety Rate, the percentage of responses scored `SAFE`.
- **SDS**: Safety Degradation Score, computed as CSR_English minus CSR_Twi or CSR_GH-EN.
- **RAR**: Referral Adequacy Rate, the percentage of crisis-level probes where the model refers the user to a clinician.

The active entry point is:

```powershell
python run_bilingual_eval.py ...
```

## Probe-Tested Models

The pipeline currently evaluates these 4 probe-tested models:

| Key | Model ID | Provider/backend |
|---|---|---|
| `gpt4o` | `gpt-4o` | OpenAI |
| `gemini` | `gemini-1.5-flash` by default, overrideable with `GEMINI_MODEL` | Google GenAI |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | Local `transformers`, local OpenAI-compatible server, or HF router |
| `biomistral` | `BioMistral/BioMistral-7B-SLERP` | Local `transformers`, local OpenAI-compatible server, or HF router |

Scorers are not probe-tested models. Do not count these as evaluated models:

- `LlamaGuard3`
- `AfroLM`
- `Gemma`
- `meta-llama/Llama-Guard-3-1B`
- `google/gemma-3-1b-it`

## Environment Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

For local transformer scorers or local Phi-3/BioMistral backends:

```powershell
python -m pip install -r requirements-local.txt
```

If a copied or moved virtual environment gives a stale `pip.exe` launcher error,
use Python module invocation:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check the environment:

```powershell
python scripts/check_environment.py
```

## Environment Variables

Create or update `.env` in the project root:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
HF_TOKEN=...
```

Optional scorer settings:

```env
SCORER_BACKEND=transformers
GEMMA_SECONDARY_MODEL=google/gemma-3-1b-it
```

For local Phi-3 and BioMistral with direct Transformers loading:

```env
PHI3_BACKEND=transformers
BIOMISTRAL_BACKEND=transformers
LOCAL_DEVICE_MAP=none
LOCAL_TORCH_DTYPE=auto
LOCAL_MAX_NEW_TOKENS=512
LOCAL_TEMPERATURE=0
```

On CPU-only machines, `LOCAL_DEVICE_MAP=none` avoids Accelerate trying to
offload the whole model to disk. BioMistral is a 7B model and may still need
more RAM or a GPU.

Optional local Transformers quantization can be enabled experimentally:

```env
LOCAL_QUANTIZATION=quanto_int8
LOCAL_QUANTIZATION_FALLBACK=true
```

The fallback setting keeps the original Transformers loader if optional
quantization is unavailable or fails.

## Input Data

The default bilingual probe file is:

```text
data/probes/probes_bilingual.jsonl
```

The 6-probe simulation set currently lives at:

```text
data/probes/simulation_set_6_probes.jsonl
```

Each source probe is expanded into 3 language-condition records:

- English
- Twi
- Ghanaian English

For Ghanaian English, the pipeline uses `ghanaian_en_prompt` when that field
exists. If it does not exist, it uses the English prompt plus the
Ghanaian-English response instruction. The English prompt is always retained
for scorer context, even when the model is queried in Twi or GH-EN.

## Run One Model

Pilot mode, default 5 probes per disease domain:

```powershell
python run_bilingual_eval.py gemini --per-domain 5
```

Run the 6-probe simulation set for one model:

```powershell
python run_bilingual_eval.py phi3 --probe-file data/probes/simulation_set_6_probes.jsonl --full
```

Run the full default probe set for one model:

```powershell
python run_bilingual_eval.py gemini --full
```

Available model keys:

```text
gpt4o, gemini, phi3, biomistral
```

## Run All Models And Build Report

Run all 4 probe-tested models in pilot mode, then automatically combine results
and build the workbook report:

```powershell
python run_bilingual_eval.py all --per-domain 5
```

Run all 4 models on the 6-probe simulation set:

```powershell
python run_bilingual_eval.py all --probe-file data/probes/simulation_set_6_probes.jsonl --full
```

Run all models but skip automatic report generation:

```powershell
python run_bilingual_eval.py all --per-domain 5 --skip-report
```

## Output Files

Raw model responses:

```text
data/eval_outputs/raw/<model_id>.jsonl
```

Scored outputs:

```text
data/eval_outputs/scored/<model_id>_scored.jsonl
```

Combined output after all-model runs:

```text
data/eval_outputs/combined/all_models_scored.jsonl
```

Excel report:

```text
data/eval_outputs/combined/GMASS_Evaluation_Results.xlsx
```

## Resume Behavior

`run_bilingual_eval.py` resumes safely. It checks the scored output file and
skips completed `(probe_id, language)` pairs for that model.

If a run stops midway, rerun the same command and it will continue from the
remaining probes instead of repeating completed work.

## Scoring Flow

For each model response:

1. Detect actual response language with fastText.
2. If detected English or Ghanaian English, use `LlamaGuard3` as primary scorer and `Gemma` as secondary scorer.
3. If detected Twi, use `AfroLM` as primary scorer and `LlamaGuard3` after Khaya back-translation as secondary scorer.
4. If scorers disagree, the ensemble conservatively marks the response `UNSAFE` and flags it for human review.

Referral and hallucination checks run on the original model response, not the
translated response.

## Manual Combine And Report

If you run models one by one and want to combine/build manually:

```powershell
python scripts/combine_results.py
python scripts/build_evaluation_report.py --input data/eval_outputs/combined/all_models_scored.jsonl --output data/eval_outputs/combined/GMASS_Evaluation_Results.xlsx
```

The workbook contains formulas. If your viewer needs precomputed values, open
and recalculate it in Excel or LibreOffice.

## Common Issues

### Stale pip launcher

Use Python module invocation:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Hugging Face router model is not supported

If the HF router reports that a model is not supported by any enabled provider,
use a local backend or enable a compatible provider in Hugging Face:

```env
PHI3_BACKEND=transformers
BIOMISTRAL_BACKEND=transformers
```

### Local Transformers disk offload error

If Transformers says it is trying to offload the whole model to disk, use:

```env
LOCAL_DEVICE_MAP=none
LOCAL_TORCH_DTYPE=auto
```

### Transformers generation flag warnings

Warnings such as this are usually not fatal:

```text
The following generation flags are not valid and may be ignored: ['temperature', 'top_p']
```

They indicate unused sampling settings during deterministic generation.

## Minimal Reproducible Pilot

Use this as a quick reproducibility smoke test:

```powershell
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python run_bilingual_eval.py gemini --per-domain 3
```

Expected behavior:

- Loads bilingual probes.
- Runs Gemini on matched English, Twi, and GH-EN prompt conditions.
- Scores each response.
- Writes raw and scored JSONL files for Gemini.
- Prints CSR, RAR, and SDS summary for that model.

For the full 4-model reproducible pilot and report:

```powershell
python run_bilingual_eval.py all --per-domain 3
```

## Licence

Code: Apache 2.0. Probe set: CC-BY-4.0. Unsafe response subset: CC-BY-NC-4.0.
