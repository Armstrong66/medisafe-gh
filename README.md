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
gmass ...
```

## Probe-Tested Models

The pipeline currently evaluates these 4 probe-tested models:

| Key | Model ID | Provider/backend |
|---|---|---|
| `gpt4o` | `gpt-4o` | OpenAI |
| `gemini` | `gemini-2.5-flash` by default, overrideable with `GEMINI_MODEL` | Google GenAI |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | Local `transformers`, local OpenAI-compatible server, or HF router |
| `biomistral` | `BioMistral/BioMistral-7B-SLERP` | Local `transformers`, local OpenAI-compatible server, or HF router |

Scorers are not probe-tested models. Do not count these as evaluated models:

- `LlamaGuard3`
- `AfroLM`
- `Gemma`
- `meta-llama/Llama-Guard-3-1B`
- `google/gemma-3-1b-it`

`gemini` can be an evaluated model key. Separately, `SCORER_BACKEND=policy_api`
means the scorer pipeline uses a hosted policy-judge runtime to execute scorer
prompts. That runtime may currently call Gemini API under the hood, but the
scorer identities remain `LlamaGuard3`, `Gemma`, and `AfroLM`.

The `gemini` key is a stable pipeline key for a Gemini Flash model, not a claim
that the historical `gemini-1.5-flash` endpoint is still available. Current
defaults use `gemini-2.5-flash`.

For extension notes, see `docs/model-and-scorer-extensibility.md`.

## Environment Setup

From the project root:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

`requirements.txt` is intentionally thin and installs the package from
`pyproject.toml`. Optional local model dependencies live in
`requirements-local.txt` / `.[local]`. The Gradio/Plotly UI dependencies live
in `requirements-app.txt` / `.[app]`.

`constraints.txt` pins the versions used by setup, Docker, and CI. Use it when
you want reproducible installs:

```powershell
python -m pip install -r requirements.txt -c constraints.txt
```

For local transformer scorers or local Phi-3/BioMistral backends:

```powershell
python -m pip install -r requirements-local.txt
```

For the local Gradio UI:

```powershell
python -m pip install -r requirements-app.txt
```

On macOS/Linux or Git Bash, the bootstrap script performs the same base setup
and installs the editable `gmass` CLI:

```bash
./setup.sh
```

On Windows PowerShell:

```powershell
.\setup.ps1
```

Install local backend, UI, or test tooling during setup with:

```bash
./setup.sh --local --app --dev
```

PowerShell equivalent:

```powershell
.\setup.ps1 -Local -App -Dev
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

Check the installed CLI:

```powershell
gmass --help
```

## Repository Structure

The repository is organized around one production path, with compatibility
wrappers kept thin:

```text
run_bilingual_eval.py  # installed as the gmass CLI
models/                # provider routing for evaluated models
scorer/                # language-aware scorer ensemble and safety checks
core/                  # config, logging, metrics, and shared utilities
probes/                # probe loading
translation/           # translation adapters
scripts/               # reports, converters, deployment helpers
configs/               # thresholds, scorer roles, model metadata
app/                   # Gradio/Hugging Face Space app files
tests/                 # regression tests for routing, scoring, metrics, reports
```

`pyproject.toml` is the dependency source of truth. The `requirements*.txt`
files are compatibility entry points for common install targets.

Legacy exploratory scripts such as `run_pilot.py`, `test_models.py`, and
`test_classifiers.py` are not the production entry point and are not installed
as package modules. Prefer `gmass` and the tested scripts in `scripts/` for
new work.

## Environment Variables

Create or update `.env` in the project root:

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
HF_TOKEN=...
```

Optional scorer settings:

```env
SCORER_BACKEND=policy_api
SCORER_POLICY_MODEL=gemini-2.5-flash
GEMMA_SECONDARY_MODEL=google/gemma-3-1b-it
```

Use `SCORER_BACKEND=transformers` only when local scorer model dependencies are
installed with `requirements-local.txt` or `.[local]`.

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
gmass gemini --per-domain 5
```

Run the 6-probe simulation set for one model:

```powershell
gmass phi3 --probe-file data/probes/simulation_set_6_probes.jsonl --full
```

Run the full default probe set for one model:

```powershell
gmass gemini --full
```

Available model keys:

```text
gpt4o, gemini, phi3, biomistral
```

## Run All Models And Build Report

Run all 4 probe-tested models in pilot mode, then automatically combine results
and build the workbook report:

```powershell
gmass all --per-domain 5
```

Run all 4 models on the 6-probe simulation set:

```powershell
gmass all --probe-file data/probes/simulation_set_6_probes.jsonl --full
```

Run all models but skip automatic report generation:

```powershell
gmass all --per-domain 5 --skip-report
```

In `gmass all`, each model runs in its own subprocess. If one provider hits a
quota, token, API, or local backend failure, the remaining model runs continue.
The command exits non-zero at the end when any model failed, but it preserves
available outputs and still attempts report generation.

## Hugging Face / Gradio Deployment

The public Gradio app lives in `app/` and imports the same pipeline modules used
by the CLI. It is suitable for an open evaluator/demo, not clinical deployment
certification.

Prepare a Space bundle from the project root:

```powershell
python scripts/prepare_hf_space.py
```

This creates:

```text
dist/hf_space/
```

Push the contents of `dist/hf_space/` to a Hugging Face Space created with:

- SDK: `Gradio`
- app file: `app.py`
- Python: 3.10+

Configure these Space secrets:

```text
OPENAI_API_KEY
GEMINI_API_KEY
HF_TOKEN
KHAYA_API_KEY
```

Recommended public CPU Space settings:

```text
SCORER_BACKEND=policy_api
GEMINI_MODEL=gemini-2.5-flash
SCORER_POLICY_MODEL=gemini-2.5-flash
```

To include precomputed benchmark charts in the Space bundle:

```powershell
python scripts/prepare_hf_space.py --include-results
```

Local app smoke test:

```powershell
python -m pip install -r requirements-app.txt -c constraints.txt
python app/app.py
```

Then open the printed local URL. The app should load without API keys; API keys
are only needed when you run model/scorer calls.

The Batch Evaluator accepts `.csv`, `.jsonl`, `.ndjson`, and `.json` uploads.
It expands mixed-language files into per-language evaluation jobs when it sees
either a `language` column with `prompt`, or language-specific prompt columns
such as `english_prompt`, `twi_prompt`, and `ghanaian_en_prompt`. Unsupported
languages are reported as skipped before model calls; if no supported probes are
found, the batch run aborts before spending provider/API compute.

### Hugging Face ZeroGPU Note

The G-MASS app does not load local GPU models by default; it uses cloud/API
model calls and CPU-side scoring helpers. If the Space is forced onto ZeroGPU,
the app includes a small `@spaces.GPU` compatibility marker so the ZeroGPU
runtime can start, but the recommended hardware for this public demo remains a
CPU Space when available.

## GitHub Codespaces Temporary Hosting

GitHub Pages cannot host this app because Pages is static and G-MASS is a
Python/Gradio server. For a temporary live demo from GitHub, use Codespaces:

1. Push this repo to GitHub.
2. Open the repo in GitHub Codespaces.
3. Wait for `.devcontainer/devcontainer.json` to install base and app deps.
4. Run:

   ```bash
   bash scripts/run_codespaces_app.sh
   ```

5. Open the forwarded port `7860`. Set port visibility to public if you need to
   share the temporary demo URL.

Codespaces URLs are suitable for pilot testing and demos while the Codespace is
running. They are not a permanent public deployment target.

## Docker

Build a reproducible base image from the project root:

```powershell
docker build -t medisafe-gh .
```

Include local Transformers backend dependencies in the image only when needed:

```powershell
docker build --build-arg INSTALL_LOCAL=true -t medisafe-gh-local .
```

Run the CLI in the container with local credentials supplied at runtime:

```powershell
docker run --rm --env-file .env -v ${PWD}/data/eval_outputs:/app/data/eval_outputs medisafe-gh --help
```

Example pilot run:

```powershell
docker run --rm --env-file .env -v ${PWD}/data/eval_outputs:/app/data/eval_outputs medisafe-gh gemini --per-domain 3
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

If the installed `gmass.exe` console launcher points at an old copied or moved
Python path, use the module entry point or recreate the virtual environment:

```powershell
.\venv\Scripts\python.exe -m run_bilingual_eval --help
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
python -m pip install -e .
gmass gemini --per-domain 3
```

Expected behavior:

- Loads bilingual probes.
- Runs Gemini on matched English, Twi, and GH-EN prompt conditions.
- Scores each response.
- Writes raw and scored JSONL files for Gemini.
- Prints CSR, RAR, and SDS summary for that model.

For the full 4-model reproducible pilot and report:

```powershell
gmass all --per-domain 3
```

---

## Citation

Please cite our work:

```bibtex
@misc{medisafe_gh2026,
  title={MediSafe-GH: A Clinical Safety Screen for Medical AI Assistants in Ghanaian Languages},
  author={Koduah, Joseph Derrick Anane Nti and Asare, Michael Asiedu and Owusu, Emmanuel and Yeboah, Benjamin Appiah},
  year={2026},
  howpublished={\url{https://github.com/Armstrong66/medisafe-gh}},
  note={Africa AI Safety Prize Competition 2026}
}
```

---


## Licence

Code: Apache 2.0. Probe set: CC-BY-4.0. Unsafe response subset: CC-BY-NC-4.0.
