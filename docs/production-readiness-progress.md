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
- Changed `.env.example` to default to `SCORER_BACKEND=policy_api`, making clear that Gemini may be the hosted execution runtime while scorer identities remain LlamaGuard3/Gemma/AfroLM.
- Added Dockerfile and `.dockerignore` for a reproducible base image that keeps credentials runtime-only.
- Updated README setup, run, smoke-test, and Docker instructions to prefer `gmass`.
- Updated `scripts/check_environment.py` to validate converter dependencies and the installed `gmass` CLI.
- Fixed a mojibake artifact in `scorer/scorer.py`.
- Fixed a mojibake artifact in the Hugging Face Space metadata and aligned the Space Gradio version with the tested app dependency.

## Remaining

- Add a lockfile or fully pinned constraints file for dependency drift control.
- Add CI smoke tests for `./setup.sh`, `gmass --help`, and a minimal report-only or mocked eval path.
- Add run manifests with commit SHA, config digest, model IDs, dependency snapshot, and artifact checksums.
- Split the current CLI into clearer subcommands such as `gmass eval`, `gmass combine`, and `gmass report` if the interface needs to grow.
