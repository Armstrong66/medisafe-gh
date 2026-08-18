Immediate plan (short-term, actionable):

1. Expand and harden hallucination/referral detectors
   - Mine historical raw scorer outputs to extract candidate hedge/referral phrases.
   - Normalize matching (lowercase, punctuation strip, Unicode NFKC, basic lemmatization).
   - Add fuzzy-matching and unit tests for paraphrase/negation cases.
   - (Medium-term) Train a light classifier on curated examples to detect hallucination/referral.

2. Implement CSV/XLSX -> JSONL converter
   - scripts/convert_probes.py: canonicalize column names, validate required fields, emit JSONL.
   - Add tests and update check_environment.py to list converter dependencies.
   - For PDFs: document limitation and provide guidance rather than implementing a universal parser.

3. Reproducible execution and CI
   - Add Dockerfile plan and a CI smoke test that runs setup.sh and a minimal eval run.
   - Capture run-manifest metadata (commit SHA, config digest, model IDs, dependency snapshot) for each evaluation.

4. Tracking & next steps
   - Create unit tests for expanded detectors and data converters.
   - Add provider budget controls work package (timeouts/retries/spend caps).
