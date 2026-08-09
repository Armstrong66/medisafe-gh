# G-MASS App Deployment Readiness Plan

Date: 2026-08-09

## Current Decision

The `app/` folder is now aligned with the current repository pipeline and can
serve as a Hugging Face Spaces deployment package after the source changes are
committed and pushed.

It is suitable for an open public evaluator/demo that calls configured model
APIs and runs the G-MASS scoring pipeline. It should not be described as a
clinical deployment certification tool.

## Aligned Interfaces

- App entrypoint: `app.py`, which launches `gmass_app.demo`.
- Core model calls: `models.router.call_model`.
- Language instruction: `models.router.build_prompt_with_language_instruction`.
- Scoring: `scorer.scorer.GMassScorer`.
- Metrics: `core.metrics.full_model_profile`.
- Real benchmark display: reads `data/eval_outputs/combined/all_models_scored.jsonl` when present.

## Model And Scorer Contract

Evaluated model keys:

- `gpt4o`
- `gemini`
- `phi3`
- `biomistral`

Scorer identities:

- LlamaGuard3
- Gemma
- AfroLM

`SCORER_BACKEND=policy_api` is a scorer runtime option. It may call Gemini API
to execute scorer prompts, but Gemini is not a scorer identity and should not
be counted as one in reports.

## Required Runtime Secrets

- `OPENAI_API_KEY` for GPT-4o.
- `GEMINI_API_KEY` for Gemini evaluated-model calls and `SCORER_BACKEND=policy_api`.
- `HF_TOKEN` for Hugging Face router/open-weight model calls.
- `KHAYA_API_KEY` if using hosted Khaya translation.

Never commit secrets into the Space repo.

## Deployment Steps

1. Commit and push the main `medisafe-gh` repository changes.
2. Create or clone the Hugging Face Space repository.
3. Copy these files from `app/` into the root of the Space repo:
   - `app.py`
   - `gmass_app.py`
   - `spaces_README.md` as `README.md`
   - `spaces_requirements.txt` as `requirements.txt`
4. Copy the main repo `configs/` directory into the Space repo so `core.config`
   loads explicit thresholds instead of fallback defaults.
5. Optionally copy `data/eval_outputs/combined/all_models_scored.jsonl` if the
   Benchmark Results tab should show precomputed real results.
6. Configure Space secrets listed above.
7. Set `SCORER_BACKEND=policy_api` for the public CPU Space unless running a GPU/local-scorer tier.
8. Push the Space repo and wait for the Gradio build.
9. Run one English probe and one Twi/GH-EN probe to confirm:
   - model calls work,
   - scorer runtime works,
   - language consistency flag appears when relevant,
   - errors are shown without stack traces.

## Not Yet Production-Hardened

- The public CPU Space should not run full high-volume production batches without provider budget caps.
- Run manifests and artifact checksums are still needed for audit-grade evaluation runs.
- CI should test the Space import path and `demo` construction.
- If actual local LlamaGuard3/Gemma/AfroLM weights are required in production, deploy a separate GPU Space or container with `.[local]` installed and `SCORER_BACKEND=transformers`.
