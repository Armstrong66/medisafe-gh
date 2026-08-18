# G-MASS production-readiness review

Date: 2026-08-04
Scope: pipeline review from bootstrap through evaluation, scoring, metrics, and output generation.

## Executive summary

The repository is much closer to a usable evaluation pipeline than it was before, and the core readiness logic is now materially stronger. The main remaining risks are not in the scoring formula alone; they are in operational reproducibility, deployment automation, and auditability.

In its current state, the project is suitable for a controlled pilot or internal benchmark run. It is not yet fully production-grade for repeatable external deployment because:

- bootstrap still relies on host-level Python state rather than a reproducible container image;
- package installation is partially automated but not fully complete for all backends;
- the CLI is usable but not packaged as a first-class installable tool;
- output artifacts are generated, but there is no strong manifest/checksum/replay workflow yet;
- provider-side failures, rate limits, and cost controls are still mostly implicit rather than governed by policy.

## Stage-by-stage review

### 1) Bootstrap and environment setup

Current state:
- `setup.sh` now creates `.env` from `.env.example` when needed, installs base dependencies with `pip`, and downloads the fasttext language-ID model.
- `scripts/check_environment.py` validates Python version, package availability, `.env` presence, and obvious placeholder values.
- `.env.example` is now tracked and provides a contributor-safe starting point.
- Resolved update, 2026-08-05: `setup.sh` now installs the editable package, exposes the `gmass` CLI, and supports explicit `--local` and `--dev` install modes.
- Resolved update, 2026-08-05: `setup.ps1` provides the same base/local/dev setup path for direct Windows PowerShell use.

What is working well:
- Fresh-clone onboarding is much better than before.
- The bootstrap path no longer silently assumes an already-prepared environment.
- Placeholder keys are now caught early.

Remaining gaps:
- Local Transformers dependencies remain opt-in through `./setup.sh --local` because they are heavier and machine-specific.
- The setup flow does not pin or lock dependencies; it relies on floating package versions.
- There is no checksum validation for the downloaded `lid.176.ftz` model.
- There is no CI bootstrap smoke test that verifies the setup script end to end on a clean environment.

### 2) CLI and orchestration logic

Current state:
- `run_bilingual_eval.py` is the main CLI entry point and uses `argparse`.
- It supports `model`, `--probe-file`, `--per-domain`, `--full`, `--delay`, and `--skip-report`.
- The `all` mode runs each model sequentially and then combines outputs.
- Resolved update, 2026-08-05: the runner is now import-safe and exposed as the installable `gmass` console command via `pyproject.toml`.

What is working well:
- The CLI is straightforward and understandable for local use.
- The `all` flow is convenient for a full run.

Remaining gaps:
- There is no explicit resume/replay manifest; reruns depend on the existing JSONL files and deduplication logic.
- The pipeline has no built-in dry-run mode, no budget cap, and no operation-level timeout policy.
- The CLI does not clearly separate “pilot”, “full evaluation”, and “report-only” modes in a production-friendly way.

### 3) Model provider integration

Current state:
- `models/router.py` supports OpenAI, Gemini, and local/open-weight backends.
- The router has some retry logic for Gemini and some optional local fallback handling for transformers.
- The code also supports a local HF router path for some inference backends.

What is working well:
- The model abstraction is reasonable and flexible.
- The provider routing is configurable from environment variables, which helps experimentation.

Remaining gaps:
- Provider timeouts, retry budgets, and cost ceilings are still mostly implicit.
- There is no central policy for “stop on model failure”, “skip and continue”, or “fail closed” for expensive providers.
- Local backends can still fail in a noisy way if the machine lacks the right packages or GPU/runtime preconditions.
- Credentials are validated ad hoc instead of via a provider-aware preflight policy.

### 4) Scoring and metrics

Current state:
- The scorer pipeline now uses response-language detection to route Twi vs English/GH-EN evaluations more correctly.
- The readiness logic is now separated into explicit gates: SDS parity signal, CSR floor, RAR target, and evaluability.
- `deploy_status` is surfaced to the user-facing outputs.

What is working well:
- The deployment-readiness logic is now much less prone to false positives from “equally bad” English/Twi safety scores.
- The code now distinguishes parity from absolute safety, which is important for deployment decisions.

Remaining gaps:
- The metrics remain dependent on the current probe set and scoring design; they should be treated as a decision aid, not the only production gate.
- The pipeline still lacks a formal canary/holdout enforcement layer and a clear review path for high-risk cases.
- There is no versioned threshold policy that records why a threshold changed and who approved it.

### 5) Outputs, reporting, and auditability

Current state:
- Raw and scored JSONL artifacts are written per model.
- `scripts/combine_results.py` combines results and prints a summary.
- The report-building path exists and can produce a workbook.

What is working well:
- The outputs are structured enough to support downstream analysis.
- The pipeline produces a combined artifact for aggregate metrics.

Remaining gaps:
- There is no formal artifact manifest or checksum bundle for the produced outputs.
- There is no immutable run metadata capture (commit SHA, config digest, prompt version, model version, dependency versions, environment snapshot).
- The current workflow is still mostly file-based and manual; it would be stronger with a declarative run manifest and an audit log.

## Current state of CLI, Docker, and package installation

### CLI behavior
- Resolved update, 2026-08-05: the package now installs a stable `gmass` console command.
- The script path remains available for local compatibility, but contributor docs now prefer `gmass ...`.
- Remaining packaging polish would be to split subcommands such as `gmass eval`, `gmass combine`, and `gmass report`; the current command preserves the existing argument contract.

### Docker / containerization
- Resolved update, 2026-08-05: a Dockerfile and `.dockerignore` now provide a reproducible base image for CLI evaluation runs.
- Credentials remain runtime-only via environment variables or `--env-file .env`; the image does not bake secrets into the build.
- Remaining production hardening would be to add CI image build/publish and dependency lock validation.

### Package installation
- The repository has `requirements.txt`, `requirements-local.txt`, and `pyproject.toml`.
- `setup.sh` is the main bootstrap path for contributors.
- Resolved update, 2026-08-05: `setup.py` now delegates to `pyproject.toml`; `pyproject.toml` defines package metadata and a `gmass` entry point.
- Resolved update, 2026-08-05: `setup.sh` installs the editable package and supports explicit `--local` and `--dev` install modes.
- Resolved update, 2026-08-05: `requirements.txt` and `requirements-local.txt` are thin compatibility files that point to the canonical `pyproject.toml` dependency sets.
- Local-model support remains opt-in because those dependencies are heavier and machine-specific.
- Because there is no lockfile, dependency drift remains possible across machines and CI environments.

## Recommended next steps (priority order)

High priority:
1. Add a lockfile and a CI smoke test that runs the bootstrap path and a minimal eval path.
2. Add run manifests with config/model/dependency fingerprints for every evaluation batch.
3. Make local backends and cloud backends install and validate through the same provider-aware preflight path.
4. Add dependency/image checksum validation for downloaded or built artifacts.

Medium priority:
5. Add explicit provider budget controls (timeouts, retries, max spend, fail-fast policy).
6. Add a stronger artifact integrity workflow (checksums, manifest, immutable outputs).
7. Split the installed CLI into clearer subcommands if the command surface grows.

## Bottom line

This repo now has a solid foundation for a pilot-grade safety evaluation workflow. It is not yet fully hardened for production-grade deployment, but the remaining work is now concentrated around lockfiles, CI smoke tests, run manifests, artifact integrity, and provider governance.

This review should be treated as a contributor-facing checkpoint rather than a final release gate. The most important next move is to make the environment and execution path reproducible before scaling the evaluation workload.

## Hallucination and referral detector limitations (EN + TWI)

Findings:
- The current HallucinationDetector in scorer/scorer.py uses a fixed list of hedge phrases (_HEDGE_PHRASES) and returns True (hallucination) when none of those hedging phrases appear and referral_flag is False. This is a pragmatic, conservative heuristic but has important limitations:
  - It relies on a small dictionary of hedge phrases and will miss hallucinations where the model expresses confidence with alternative phrasing not in the list.
  - Negation, punctuation, or paraphrases can evade the simple substring checks (e.g., "I do not think" vs "I can't say").
  - Confidence is treated as a boolean-like signal derived from surface text rather than any calibrated model confidence, so false negatives are possible.

- The ReferralDetector similarly uses a small set of English and Twi keyword phrases. It detects explicit referral language but will miss more subtle or paraphrased referral recommendations, and it is sensitive to surface-level variations (typos, punctuation, different word order).

Feasibility of extracting detector keywords from judge scorers:
- It is possible to extract common phrases/indicators from the judge scorers' raw outputs (e.g., the primary/secondary scorers' 'raw_output' or 'categories' fields) to build a larger, empirically-derived keyword set. Minimal approach:
  - Aggregate the raw outputs and categories from previous scored runs (the 'raw_output' fields in scored JSONL), identify high-confidence UNSAFE cases flagged as hallucination or referral failures, and extract n-grams and phrase patterns.
  - Use frequency thresholds and a small manual review pass to curate a candidate keyword list for both English and Twi.
- Limitations of this minimal approach:
  - It is still dictionary/heuristic-based and will not generalize to many paraphrases unless the extraction is made more sophisticated (e.g., fuzzy matching, lemmatization, or light semantic clustering).
  - The approach depends on the available historical judged outputs; if those are sparse or biased, the extracted dictionary will inherit that bias.

Recommended immediate actions (hallucination/referral detectors):
1. Expand the hedge-phrases and referral keyword lists by mining the raw scorer outputs from past runs (build a candidate list, then curate).
2. Replace simple substring checks with normalized matching (lowercasing, punctuation stripping, Unicode normalization, basic lemmatization) and support fuzzy matching thresholds for tolerant detection.
3. Add unit tests that feed paraphrased and negated forms to the detectors to quantify false-negative risk.
4. Long-term: build a small classifier (light-weight transformer or logistic regression on embedding features) trained on curated examples to detect hallucination/referral signals more robustly across paraphrase variants and languages.

## Extensibility to additional diseases and language domains

- The pipeline already reads domains from configs/gmass_config.yaml via core/config.py and discovers domains from the data when building domain-level metrics. This design makes adding new disease domains straightforward: update the config or introduce new probe sets containing the new domains.
- Language extensibility is supported via response-language detection and the scorer role policy loaded from `configs/gmass_config.yaml`. Current default and Twi role replacements are config-driven; adding a new language bucket still requires extending the scorer policy mapping and tests. Adding a new language requires:
  - Extending fasttext detection (if fasttext supports it) or integrating another language-ID model that covers the target language.
  - Providing a primary scorer or a ruleset for that language (e.g., a native model like AfroLM for Twi) and deciding the routing/translation strategy (native scoring vs back-translation + English scorer).
- Action items:
  - Document the steps to add a new disease domain and to add a new language in the repo (configs, scorer mapping, probe format expectations).
  - Add CI checks that flag missing scorer implementations when a new language is present in probes.

## Data-ingestion: CSV / XLSX / PDF → JSONL conversion

- Current expectation: probe input is JSONL with canonical fields (probe_id, disease_domain, failure_category, english_prompt, twi_prompt, ...).
- Feasibility:
  - CSV / XLSX: straightforward to convert reliably to JSONL as long as the spreadsheet preserves the required fields and text encoding. Implement a converter script that:
    - Validates required columns, normalizes column names to canonical field names, and emits JSONL lines with proper escaping and UTF-8 normalization.
    - Runs sanity checks (no empty probe_id, limited prompt length, required fields present).
  - PDF: problematic to convert reliably. While OCR and heuristic extraction can sometimes yield usable text, structured fields (probe_id, domain, category) are often lost unless the PDFs follow a strict template. Converting arbitrary PDFs risks distorting text and losing structure.

- Recommended approach:
  1. Implement a CSV/XLSX converter (scripts/convert_probes.py) that:
     - Reads CSV/XLSX, maps columns to canonical fields, performs Unicode normalization, and writes JSONL.
     - Emits a validation report highlighting rows with missing fields.
  2. For PDF inputs, do not attempt a universal converter. Instead:
     - Provide guidance in docs for contributors with PDF-only data: prefer exporting to CSV/XLSX first, or provide a templated spreadsheet.
     - Optionally implement a limited PDF parser for strictly templated PDFs as a separate non-default utility, and flag caution about noise.

Immediate action items (data ingestion):
- Add scripts/convert_probes.py for CSV/XLSX → JSONL conversion and unit tests.
- Update scripts/check_environment.py to mention the converter and any additional dependencies (pandas, openpyxl).
- Add documentation in docs/ describing acceptable input formats and a recommended contributor workflow for PDF-origin data.

## Documentation updates and tracking

- The above limitations and recommended actions will be added to this review document and to plan.md. Add todos to the session DB to track the following prioritized work items:
  - Expand and harden hallucination/referral detectors (short-term + medium-term classifier plan).
  - Implement CSV/XLSX → JSONL converter and tests.
  - Add CI smoke test for setup.sh, package installation, and the Docker image build.


