# Model And Scorer Extensibility

Date: 2026-08-09

Status update: 2026-08-12

## Tested Models

The evaluated model surface is intentionally small and easy to extend:

- `models/router.py` owns provider calls and the `MODEL_FUNCTIONS` dispatch map.
- `run_bilingual_eval.py` owns `MODEL_ID_MAP` and the accepted CLI model keys.
- `configs/models.yaml` documents the public model lineup and API key needs.
- The Gradio app reads the same router constants and model keys.

To add a tested model, add a provider wrapper in `models/router.py`, register it
in `MODEL_FUNCTIONS`, add its display/model ID to `run_bilingual_eval.py` and
`configs/models.yaml`, then add a smoke or unit test for the new route.

The `gemini` key is kept as a stable compatibility key for a Gemini Flash
evaluation target. Current defaults use `gemini-2.5-flash`; the key should not
be read as a commitment to the retired historical `gemini-1.5-flash` endpoint.

## Scorers And Judges

The current scorer identities are:

- `LlamaGuard3` for English and Ghanaian-English primary scoring.
- `Gemma` for English and Ghanaian-English secondary validation.
- `AfroLM` for native Twi primary scoring.
- `LlamaGuard3` again for detected-Twi secondary validation after Khaya
  back-translation.

Scorer runtime backends are already replaceable:

- `SCORER_BACKEND=policy_api` uses hosted policy-judge execution.
- `SCORER_BACKEND=transformers` loads configured local/open-weight scorer
  models where supported.
- `LLAMAGUARD_MODEL_ID`, `GEMMA_SECONDARY_MODEL`, and related env vars replace
  concrete scorer model IDs without code changes.

The scorer architecture is moderately extensible, but not yet a full plugin
system. Current primary/secondary role routing is declarative for the supported
language buckets:

- `configs/gmass_config.yaml` declares the default primary/secondary scorer
  roles and the Twi-specific primary/secondary scorer roles.
- `core/config.py` loads those scorer role names.
- `GMassScorer` validates the configured names against the supported scorer
  implementations, then builds the primary/secondary ensemble from that policy.

Replacing a current scorer role is now config-driven for `LlamaGuard3`, `Gemma`,
and `AfroLM`. Adding a brand-new scorer implementation, a new language bucket,
or a different voting policy still requires code changes in `scorer/scorer.py`
plus tests.

Recommended next improvement: promote the language-bucket mapping itself to a
YAML list or mapping, so new languages can be added without editing
`SCORER_POLICY_BY_LANGUAGE`.
