# Changelog

All notable changes to G-MASS (Ghana Medical AI Safety Screen) are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned (Roadmap v1.2.0 - v2.0.0)
- Full 300-probe × 4-model evaluation dataset release on Hugging Face.
- Ga language extension (`GMASS-probe-set-v1.1`).
- Maternal health domain extension (50 new probes, `GMASS-probe-set-v1.2`).
- Framework protocol layer (`ModelCaller`, `SafetyScorer`, `GMassRegistry`) for v2.0.0.

---

## [1.1.0] - 2026-08-18

### Added
- **Gradio Public Web & Batch Evaluator**: Interactive single-probe evaluator and batch probe processor with support for CSV, JSON, JSONL, and NDJSON file uploads.
- **Auto-Parsing for Batch Probes**: Flexible schema detection supporting mixed-language rows, language-specific prompt columns (`english_prompt`, `twi_prompt`, `ghanaian_en_prompt`, `source_standard_english`, `final_approved_twi`, etc.), and automatic fallback probe IDs.
- **Public Metrics Layer (`scripts/export_public_metrics.py`)**: Automatic generation of public-safe aggregated benchmark metrics (`data/public_metrics/benchmark_summary.json` and Markdown tables) allowing open hosting of CSR, SDS, RAR, and domain safety metrics on GitHub and Hugging Face without exposing raw response text.
- **Hugging Face Space Bundle Builder**: `scripts/prepare_hf_space.py` packaging public metrics, Gradio app, and pipeline modules for deployment.
- **Scorer Role Configuration**: Config-driven primary and secondary judge role definitions in `configs/gmass_config.yaml` and `core/config.py`.

### Changed
- **Gemini Evaluated Model Default**: Updated stable evaluated model default to `gemini-2.5-flash`.
- **Packaging**: Made `medisafe-gh` fully pip-installable with console entry point `gmass` (`run_bilingual_eval:main`).
- **Dependency Extras**: Split `requirements.txt` into lightweight base, `requirements-local.txt` (`.[local]` for Transformers/PyTorch), and `requirements-app.txt` (`.[app]` for Gradio/Plotly) with pinned `constraints.txt`.

### Fixed
- **Mojibake Elimination**: Replaced multi-byte separators in `core/logger.py` with clean ASCII hyphens to ensure consistent rendering in Windows PowerShell, Linux CI, and cloud logs.
- **Log Noise Reduction**: Lowered expected empty-slice language metrics from `WARNING` to `DEBUG` in `core/metrics.py`.
- **JSONL Missing-File Warnings**: Added `warn_missing=False` flag in `core/utils.py` and silenced false warnings on fresh evaluation runs and optional artifacts.
- **Config Key Defaults**: Replaced strict dictionary indexing with `.get()` defaults in `core/config.py` to prevent false fallback warnings on optional configuration keys.

---

## [1.0.1] - 2026-08-05

### Fixed
- **Referral & Hallucination Detector Hardening**: Unicode NFKD normalization, punctuation-insensitive matching, and typo-tolerant phrase checks.
- **Twi Referral Phrases**: Expanded keyword anchors for Twi medical emergency and clinic referral expressions.
- **Cross-Platform Setup**: Added `setup.ps1` for native Windows PowerShell setups and updated `setup.sh`.
- **Environment Doctor**: Updated `scripts/check_environment.py` with CLI checks and converter validation.

---

## [1.0.0] - 2026-06-30

### Initial Release
- Africa AI Safety Prize Competition 2026 submission (3rd Place, Track II).
- Bilingual evaluation pipeline covering English, Ghanaian English, and Twi across 6 disease domains.
- Ensemble safety scoring: LlamaGuard3-1B (English primary), AfroLM (Twi primary), Gemma3-1B (cross-validator), and fastText LID.
- Core safety metrics: Clinical Safety Rate (CSR), Safety Degradation Score (SDS), and Referral Adequacy Rate (RAR).
