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

# G-MASS: Ghana Medical AI Safety Screen
**MediSafe-GH · Track II Africa AI Safety Prize · KNUST Bioinstrumentation & Medical Imaging Laboratory**

G-MASS evaluates whether AI health assistants respond safely to clinical queries across **English**, **Ghanaian English**, and **Twi**.

---

## 🚀 How to Use the Interface

1. **Single Probe Evaluation**: Enter a medical question in English, Ghanaian English, or Twi, choose your target model, and evaluate for immediate safety verdicts, language detection, and clinical referral adequacy.
2. **Batch Evaluation**: Upload `.jsonl`, `.csv`, `.ndjson`, or `.json` datasets to run evaluations across entire probe sets and download scored CSV reports.
3. **Benchmark Results**: Inspect empirical Clinical Safety Rates (CSR) and Cross-Lingual Safety Degradation Scores (SDS).
4. **Settings & Compute Tiers**: Enter custom session API keys, adjust SDS deployment thresholds, or toggle between judge compute tiers.
5. **Community & Issue Tracker**: Submit clinical safety hazard reports, flag false positives or Twi dialect nuances, and open direct GitHub Issues or Pull Requests.
6. **Contact & Support**: Reach out to the KNUST research team directly at `biomedicaltechnologieslab@gmail.com`.

---

## ⚙️ Compute Tiers (Vision §2)

G-MASS supports adaptive compute scaling:

- **Tier 1 (Nano)**: CPU-only FastText + lightweight rule heuristics (~0.3s/probe).
- **Tier 2 (Standard - Default)**: LlamaGuard3-1B + AfroLM ensemble (~1–2s/probe).
- **Tier 3 (Heavy)**: 16GB+ VRAM GPU, full LlamaGuard3-8B / Gemma3-7B research ensemble.
- **Tier 4 (API-only)**: Zero local compute, hosted cloud policy judge.

---

## 🔑 Required & Optional API Keys

- `GEMINI_API_KEY`: Gemini 2.5 Flash evaluation and hosted policy judge.
- `OPENAI_API_KEY`: GPT-4o / GPT-4o mini evaluations.
- `HF_TOKEN`: Hugging Face router / open-weight models (Phi-3, BioMistral).
- `KHAYA_API_KEY`: Real-time Khaya / GhanaNLP translation.

> **Note**: Custom keys can be entered directly in the **Settings** tab for individual sessions without exposing secrets.

---

## 📬 Contact & Support

- **Email**: [biomedicaltechnologieslab@gmail.com](mailto:biomedicaltechnologieslab@gmail.com)
- **GitHub**: [Armstrong66/medisafe-gh](https://github.com/Armstrong66/medisafe-gh)
- **Space**: [BioinstLab/gmass-demo](https://huggingface.co/spaces/BioinstLab/gmass-demo)
- **Institution**: Bioinstrumentation & Medical Imaging Laboratory, Department of Biomedical Engineering, KNUST, Kumasi, Ghana.
