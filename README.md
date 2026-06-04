# MediSafe-GH: G-MASS Evaluation Protocol

## Repo structure
medisafe-gh/                        ← GitHub repo root
│
├── medisafe_gh/                    ← Python package (importable)
│   ├── __init__.py                 ← exposes top-level API
│   ├── core/
│   │   ├── config.py               ← YAML loader
│   │   ├── logger.py               ← nohup-safe rotating logger
│   │   ├── metrics.py              ← CSR, SDS, RAR (single source of truth)
│   │   └── utils.py                ← I/O, caching, env detection
│   ├── probes/
│   │   ├── builder.py              ← AfriMed-QA seeding + example probes ✓
│   │   └── loader.py               ← load/filter probe JSONL files [TODO]
│   ├── scoring/
│   │   └── scorer.py               ← LlamaGuard3 + RoBERTa ensemble ✓
│   ├── audio/
│   │   └── pipeline.py             ← Whisper ASR + Khaya TTS [TODO - Phase 3]
│   └── cli.py                      ← `gmass` entry point [TODO]
│
├── configs/
│   ├── gmass_config.yaml           ← domains, thresholds, languages [TODO]
│   └── models.yaml                 ← model IDs, API vs local [TODO]
│
├── data/
│   ├── probes/                     ← probes_en.jsonl, probes_twi.jsonl, probes_gh_en.jsonl
│   ├── eval_outputs/raw/           ← one JSONL per model (raw responses)
│   ├── eval_outputs/scored/        ← one JSONL per model (labelled)
│   ├── simulation/                 ← 10 clinical scenario transcripts
│   └── audio/                      ← Khaya TTS .wav files
│
├── notebooks/
│   ├── 01_pilot_eval.ipynb         ← 30-probe pilot [your next task]
│   ├── 02_full_eval.ipynb
│   ├── 03_simulation.ipynb
│   └── 04_results_analysis.ipynb
│
├── scripts/
│   └── build_probe_set.py          ← runs builder.py end-to-end
│
├── tests/
│   ├── test_metrics.py
│   └── test_scorer.py
│
├── logs/                           ← gmass_eval.log (gitignored)
├── pyproject.toml                  ✓
├── requirements.txt
├── .env.example                    ← template (real .env gitignored)
├── .gitignore
└── README.md

## Overall Pipeline (End-to-End) 
### NB: audio generation workflow might be entirely dropped to constrain this work
AfriMed-QA (clinical knowledge)
        │
        ▼
[builder.py] Team B drafts 300 English probes (6 domains × 3 categories)
        │
        ▼
[Khaya API / GhanaNLP model] Team C generates Twi drafts
        │
        ▼
Human validators approve Twi probes → probes_twi.jsonl
        │
        ▼
[Khaya TTS] generates .wav audio versions of Twi + GH-EN probes
        │
        ▼
[evaluate.py / notebooks] call 5 model APIs/local inference
  → raw responses saved → data/eval_outputs/raw/<model>.jsonl
        │
        ▼
[scorer.py] for each response:
  ├─ if Twi response → translate to English first (see below)
  ├─ LlamaGuard3 scores (probe_en, response_en) → label
  ├─ RoBERTa cross-validates response_en → label
  ├─ Ensemble → final safety_label + disagreement flag
  ├─ ReferralDetector → referral_flag (on original Twi text)
  └─ HallucinationDetector → hallucination_flag
        │
        ▼
scored JSONL → data/eval_outputs/scored/<model>_scored.jsonl
        │
        ▼
[metrics.py] compute CSR, SDS, RAR per model per language
        │
        ▼
Simulation (10 scenarios × 5 models) + full results report
        │
        ▼
HuggingFace dataset upload + GitHub release