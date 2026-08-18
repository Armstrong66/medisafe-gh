---
title: G-MASS - Ghana Medical AI Safety Screen
emoji: 🏥
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - medical-ai
  - ai-safety
  - ghana
  - twi
  - african-languages
  - llm-evaluation
  - health
  - nlp
short_description: Medical AI safety eval for Ghanaian languages
---

# G-MASS - Ghana Medical AI Safety Screen

G-MASS tests whether AI health assistants respond safely to medical queries in
English, Ghanaian English, and Twi.

## What This Space Runs

- Evaluated model keys: `gpt4o`, `gemini`, `phi3`, `biomistral`.
- Scorer identities: LlamaGuard3, Gemma, and AfroLM.
- `SCORER_BACKEND=policy_api` may use Gemini API as the hosted scorer runtime,
  but Gemini is not counted as a scorer identity.
- Batch evaluation accepts `.csv`, `.jsonl`, `.ndjson`, and `.json` uploads.
  Files with `language` or language-specific prompt columns are expanded into
  per-language evaluation jobs; unsupported languages are skipped before model
  calls and reported in the output.
- Benchmark charts load real combined results when available; no placeholder
  benchmark numbers are displayed.

## Required Secrets

- `OPENAI_API_KEY` for GPT-4o.
- `GEMINI_API_KEY` for Gemini and `SCORER_BACKEND=policy_api`.
- `HF_TOKEN` for Hugging Face router/open-weight models.
- `KHAYA_API_KEY` when using hosted Khaya translation.

Outputs are evaluation signals, not clinical deployment certification.

## Runtime Note

This demo uses cloud/API calls and CPU-side orchestration by default. It includes
a small ZeroGPU compatibility marker for Spaces that are forced onto ZeroGPU,
but CPU hardware is the intended runtime when available.
