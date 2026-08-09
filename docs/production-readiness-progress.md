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
- Added a repeatable Hugging Face Space bundle builder and documented the Gradio deployment path.
- Reduced the packaged Python surface to the active `run_bilingual_eval`/`gmass` entry point; legacy exploratory root scripts remain in-repo but are not installed as package modules.
- Hardened `gmass all` so an API/quota/token/backend failure in one model run does not stop the remaining available models; partial failures are reported after all possible runs finish.

## Remaining

- Add run manifests with commit SHA, config digest, model IDs, dependency snapshot, and artifact checksums.
- Split the current CLI into clearer subcommands such as `gmass eval`, `gmass combine`, and `gmass report` if the interface needs to grow.
