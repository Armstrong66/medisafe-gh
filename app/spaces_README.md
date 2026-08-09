---
title: G-MASS - Ghana Medical AI Safety Screen
emoji: 🏥
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
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
short_description: Cross-lingual safety evaluation for medical AI in Ghanaian languages
---

# G-MASS - Ghana Medical AI Safety Screen

G-MASS tests whether AI health assistants respond safely to medical queries in
English, Ghanaian English, and Twi.

## What This Space Runs

- Evaluated model keys: `gpt4o`, `gemini`, `phi3`, `biomistral`.
- Scorer identities: LlamaGuard3, Gemma, and AfroLM.
- `SCORER_BACKEND=policy_api` may use Gemini API as the hosted scorer runtime,
  but Gemini is not counted as a scorer identity.
- Benchmark charts load real combined results when available; no placeholder
  benchmark numbers are displayed.

## Required Secrets

- `OPENAI_API_KEY` for GPT-4o.
- `GEMINI_API_KEY` for Gemini and `SCORER_BACKEND=policy_api`.
- `HF_TOKEN` for Hugging Face router/open-weight models.
- `KHAYA_API_KEY` when using hosted Khaya translation.

Outputs are evaluation signals, not clinical deployment certification.
