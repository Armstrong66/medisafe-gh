# Production Readiness Progress Log

Date: 2026-08-05

## Completed

- Hardened referral and hallucination detector matching with Unicode normalization, punctuation-insensitive matching, and typo-tolerant phrase checks.
- Added detector regression tests for referral punctuation/case variants, minor typos, Twi referral variants, and uncertainty hedging.
- Refactored `run_bilingual_eval.py` into an import-safe `main()` function so it can be exposed as a package CLI.
- Added the installable `gmass` console command through `pyproject.toml`.
- Made `scripts/` installable so packaged runs can still combine results and build reports.
- Updated `setup.sh` to install the editable package, expose `gmass`, and support explicit `--local` and `--dev` setup modes.
- Added `setup.ps1` for direct Windows PowerShell setup with matching `-Local` and `-Dev` modes.
- Made `pyproject.toml` the canonical dependency source; `requirements.txt` and `requirements-local.txt` are now thin compatibility entry points.
- Added `sentencepiece` to the local Transformers dependency extra so tokenizers required by local/open-weight backends install reproducibly.
- Added an `app` dependency extra and `requirements-app.txt` for the Gradio/Plotly UI.
- Updated `setup.sh` and `setup.ps1` with explicit `--app` / `-App` install modes and removed quiet pip installs so setup progress remains visible.
- Added `constraints.txt` and wired setup/Docker installs through it for pinned dependency resolution.
- Added GitHub Actions smoke coverage for `setup.sh --app`, `setup.ps1 -App`, `gmass --help`, app import, `pip check`, and Docker build.
- Changed `.env.example` to default to `SCORER_BACKEND=policy_api`, making clear that Gemini may be the hosted execution runtime while scorer identities remain LlamaGuard3/Gemma/AfroLM.
- Added Dockerfile and `.dockerignore` for a reproducible base image that keeps credentials runtime-only.
- Updated README setup, run, smoke-test, and Docker instructions to prefer `gmass`.
- Updated `scripts/check_environment.py` to validate converter dependencies and the installed `gmass` CLI.
- Fixed a mojibake artifact in `scorer/scorer.py`.
- Fixed a mojibake artifact in the Hugging Face Space metadata and aligned the Space Gradio version with the tested app dependency.
- Replaced stale Gemini 1.5 defaults with `gemini-2.5-flash` and kept `gemini` as the stable evaluated-model key for Gemini Flash.
- Added model/scorer extensibility notes documenting which replacements are config-only and which still require code changes.
- Moved current scorer role routing into `configs/gmass_config.yaml`/`core.config` so supported primary and secondary judge role replacements are config-driven and validated at scorer startup.
- Added a repeatable Hugging Face Space bundle builder and documented the Gradio deployment path.
- Reduced the packaged Python surface to the active `run_bilingual_eval`/`gmass` entry point; legacy exploratory root scripts remain in-repo but are not installed as package modules.
- Hardened `gmass all` so an API/quota/token/backend failure in one model run does not stop the remaining available models; partial failures are reported after all possible runs finish.
- Hardened the Gradio batch evaluator to accept CSV/JSON/JSONL uploads, expand mixed-language files into per-language jobs, skip unsupported languages before model calls, and abort early when no supported probes are present.
- Fixed log formatting in `core/logger.py` to use clean ASCII separators, eliminating mojibake dash artifacts in log lines across PowerShell, CI, and HF Space logs.
- Lowered empty-data metric messages in `core/metrics.py` (`compute_csr`, `compute_rar`) from warning to debug level so expected partial slices or UI imports do not pollute warning logs.
- Added `warn_missing` flag to `core/utils.py:load_jsonl()` and silenced false missing-file warnings in `load_completed_ids()`, `run_bilingual_eval.py`, and `app/gmass_app.py` for normal fresh runs and uncomputed optional artifacts.
- Hardened `core/config.py` with safe `.get()` defaults for config keys to prevent spurious `KeyError` config warning fallbacks on missing optional fields.
- Fixed invalid escape sequence in `core/utils.py` docstring.
- Verified test suite passes cleanly (108 passed, 1 skipped) and `scripts/check_environment.py` passes all checks.
- Performed commit & versioning audit in accordance with `GMASS_CODING_ASSISTANT_GUIDE.md` and `GMASS_Versioning_Roadmap.md`.
- Created root `CHANGELOG.md` following Keep a Changelog and Semantic Versioning (v1.0.0, v1.0.1, v1.1.0).
- Bumped project version to `1.1.0` in `pyproject.toml` and `app/gmass_app.py`.
- Expanded Gradio Batch Evaluator schema auto-parsing to accept and parse `.jsonl`, `.csv`, `.json`, and `.ndjson` files with automatic probe ID assignment and extended prompt column aliases (`source_standard_english`, `final_approved_twi`, `final_approved_ghanaian_english`, etc.).
- Created public metrics exporter `scripts/export_public_metrics.py` generating `data/public_metrics/benchmark_summary.json` and Markdown tables containing aggregate CSR/SDS/RAR metrics without exposing raw response text.
- Wired Gradio "Benchmark Results" tab to load directly from `data/public_metrics/benchmark_summary.json`.
- Cleaned and updated `.gitignore` to strictly exclude bytecode caches (`*.pyc`, `__pycache__`), logs, build outputs, and stray root files.
- Untracked all stale `.pyc` and log files from git index, staging clean deletions for the next commit.
- Documented local open-weight Transformers evaluation protocol (`phi3`, `biomistral`, and local ensemble scoring).
- Added `gmass --version` CLI flag returning `G-MASS v1.1.0`.
- Standardized BibTeX citation format in `README.md`.
- Documented Git unrelated-histories resolution for merging `dev/medisafe-gh-v2` into `main`.
- Successfully merged `dev/medisafe-gh-v2` into `main` with resolved unrelated histories, verified test suite (110 passed), tagged releases `v1.0.0` and `v1.1.0`, and pushed to remote `origin/main`.
- Added dynamic dataset autodiscovery engine (`auto_discover_dataset_metadata()`) in `core/config.py` enabling model-agnostic, language-agnostic, and domain-agnostic probe evaluation without hardcoded assumptions.
- Added pre-flight configuration audit engine (`validate_setup_and_configs()`) with structured warning/error logging to catch miscalibrations or missing optional credentials before execution.
- Updated Gradio UI Batch Evaluator labels to explicitly emphasize native `.jsonl` and `.csv` bilingual dataset parsing.
- Resolved Windows/Linux runner PATH resolution in `.github/workflows/smoke.yml`, `setup.ps1`, and `setup.sh` so `gmass` is discovered on all CI runners.
- Synchronized `dev/medisafe-gh-v2` and `main` branches and updated `v1.1.0` tag.
- Implemented Compute-Tiered Judge System (`compute_tier: auto | nano | standard | heavy | api`) in `configs/gmass_config.yaml` and `core/config.py:resolve_compute_tier()` (Vision §2).
- Added input security sanitization and prompt injection defense in `core/utils.py:validate_probe_input()` (Vision §9).
- Created 30-probe canary dataset `data/probes/canary_30.jsonl` and safety drift detection monitor `scripts/monitor_drift.py` logging to `data/drift_log.jsonl` (Vision §4).
- Added GitHub Actions automated canary safety gate workflow in `.github/workflows/gmass_gate.yml` (Vision §5).
- Added Settings & Personalization panel to Gradio UI (`app/gmass_app.py`) for custom API keys, SDS threshold adjustment, and compute tier overrides (Vision §7).
- Added one-command research reproducibility package (`gmass reproduce`) in `run_bilingual_eval.py` (Vision §10).
- Published Datasheet for Datasets in `docs/DATASHEET.md` and Model Card in `docs/MODEL_CARD.md` (Vision §10).
- Added comprehensive unit test suite in `tests/test_enterprise_scaling.py` with full test suite passing (115 passed, 1 skipped).
- Implemented client-side Dark/Light mode theme toggle script and responsive theme switching in Gradio UI (`app/gmass_app.py`).
- Added Community Feedback & Issue Tracker tab with color-coded urgency levels (🔴 Critical, 🟠 High, 🟡 Medium, 🔵 Low) and direct GitHub Issue / PR links.
- Added live public community feed backed by `data/community_feedback.jsonl`.
- Expanded About & User Guide with comprehensive usage instructions, API keys security guide, compute tier breakdowns (Tiers 1-4), metric glossary (CSR, SDS, RAR), and version tags history.
- Added Contact & Support tab with institutional affiliations, GitHub, LinkedIn, Hugging Face links, and contact email (`biomedicaltechnologieslab@gmail.com`).
- Updated `app/spaces_README.md` and added unit test coverage in `tests/test_app_batch_loader.py` (116 passed, 1 skipped).

## Remaining & Versioned Roadmap

### Planned for v1.2.0 (Near-term Pipeline Upgrades & Reproducibility)
- [ ] **Run Manifest Engine**: Capture run metadata (git commit SHA, YAML config hash, model identifiers, pip dependency snapshot, input probe checksums) in `outputs/manifest.json`.
- [ ] **CLI Subcommand Refactoring**: Expose structured CLI commands (`gmass eval`, `gmass score`, `gmass combine`, `gmass report`, `gmass export-metrics`).
- [ ] **Hybrid Semantic Detectors**: Integrate sentence embeddings (`all-MiniLM-L6-v2` and multilingual anchors) for paraphrased referral and subtle hallucination detection.
- [ ] **Ga Language Extension (`GMASS-probe-set-v1.1`)**: Add Ga probe loading and LID routing rules.
- [ ] **Interactive Failure Drill-down in Gradio**: Add modal/table drill-down for failed clinical probes with filter by disease domain and failure category.

### Planned for v2.0.0 (Extensible Framework & Multimodal Safety)
- [ ] **Framework Protocol Layer**: Formalize `ModelCaller`, `SafetyScorer`, and `GMassRegistry` classes for custom user models and external scorer backends.
- [ ] **Multi-turn Clinical Simulation**: Evaluate conversational drift and safety degradation across multi-turn patient-doctor interactions.
- [ ] **Ghana Health Service (GHS) Triage Level Scoring**: Classify referrals by facility tier (CHPS Compound vs District Hospital vs Regional/Teaching Hospital).
- [ ] **Audio/Voice Screen**: Evaluate transcribed voice notes (Whisper ASR + Khaya) directly for low-literacy clinical accessibility.
