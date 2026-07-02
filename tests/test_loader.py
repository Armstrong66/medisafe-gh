"""
tests/test_loader.py — Unit tests for probes/loader.py
Owner: D  |  MediSafe-GH · Africa AI Safety Prize 2026

Covers §1 of GMASS_Team_Clarifications.md — draft/validated translation
resolution must never silently overwrite either field, and must correctly
prefer the validated version when present.

Run with: pytest tests/test_loader.py -v
"""

import pytest
from probes.loader import (
    resolve_twi_prompt,
    translation_correction_rate,
    expand_bilingual_probes,
)


# ── resolve_twi_prompt (§1) ──────────────────────────────────────────────────────

class TestResolveTwiPrompt:

    def test_prefers_validated_over_draft(self):
        probe = {
            "probe_id": "GH-0001",
            "prompt_twi_draft": "Raw machine translation",
            "prompt_twi_validated": "Human-corrected version",
        }
        text, status = resolve_twi_prompt(probe)
        assert text == "Human-corrected version"
        assert status == "validated"

    def test_falls_back_to_draft_when_no_validation_yet(self):
        probe = {
            "probe_id": "GH-0001",
            "prompt_twi_draft": "Raw machine translation",
            "prompt_twi_validated": None,
        }
        text, status = resolve_twi_prompt(probe)
        assert text == "Raw machine translation"
        assert status == "draft_unreviewed"

    def test_falls_back_to_draft_when_validated_field_missing_entirely(self):
        probe = {"probe_id": "GH-0001", "prompt_twi_draft": "Raw machine translation"}
        text, status = resolve_twi_prompt(probe)
        assert text == "Raw machine translation"
        assert status == "draft_unreviewed"

    def test_backward_compat_with_legacy_flat_field(self):
        # Older schema (GMASS_150-A_probes_twi.jsonl) — only has twi_prompt
        probe = {"probe_id": "GH-0001", "twi_prompt": "Legacy flat Twi text"}
        text, status = resolve_twi_prompt(probe)
        assert text == "Legacy flat Twi text"
        assert status == "legacy_flat"

    def test_raises_when_no_twi_text_available(self):
        probe = {"probe_id": "GH-0001"}
        with pytest.raises(KeyError):
            resolve_twi_prompt(probe)

    def test_empty_string_validated_falls_back_to_draft(self):
        # Empty string is falsy — should not be treated as "validated present"
        probe = {
            "probe_id": "GH-0001",
            "prompt_twi_draft": "Draft text",
            "prompt_twi_validated": "",
        }
        text, status = resolve_twi_prompt(probe)
        assert text == "Draft text"
        assert status == "draft_unreviewed"


# ── translation_correction_rate (§1 methodology data point) ────────────────────

class TestTranslationCorrectionRate:

    def test_counts_corrected_vs_unchanged(self):
        probes = [
            {"prompt_twi_draft": "A", "prompt_twi_validated": "A corrected"},  # corrected
            {"prompt_twi_draft": "B", "prompt_twi_validated": "B"},            # unchanged
            {"prompt_twi_draft": "C", "prompt_twi_validated": "C corrected"}, # corrected
        ]
        result = translation_correction_rate(probes)
        assert result["total_with_draft_and_validation"] == 3
        assert result["corrected_count"] == 2
        assert result["unchanged_count"] == 1
        assert result["correction_rate_pct"] == pytest.approx(66.67, abs=0.01)

    def test_counts_unreviewed_separately(self):
        probes = [
            {"prompt_twi_draft": "A", "prompt_twi_validated": "A corrected"},
            {"prompt_twi_draft": "B", "prompt_twi_validated": None},  # not yet reviewed
        ]
        result = translation_correction_rate(probes)
        assert result["still_unreviewed_count"] == 1
        assert result["total_with_draft_and_validation"] == 1  # unreviewed excluded from denominator

    def test_legacy_probes_excluded_from_denominator(self):
        probes = [
            {"twi_prompt": "Legacy only, no draft field at all"},
        ]
        result = translation_correction_rate(probes)
        assert result["total_with_draft_and_validation"] == 0
        assert result["correction_rate_pct"] == 0.0


# ── expand_bilingual_probes integration with resolve_twi_prompt ────────────────

class TestExpandBilingualProbesIntegration:

    def test_expanded_twi_record_uses_validated_when_present(self):
        bilingual = [{
            "probe_id": "GH-0001",
            "disease_domain": "Malaria",
            "failure_category": "Harmful Advice Request",
            "english_prompt": "English text",
            "prompt_twi_draft": "Draft Twi",
            "prompt_twi_validated": "Validated Twi",
        }]
        result = expand_bilingual_probes(bilingual)
        twi_record = result["twi"][0]
        assert twi_record["prompt"] == "Validated Twi"
        assert twi_record["translation_status"] == "validated"
        # Both source fields preserved, per §1 — never overwritten
        assert twi_record["prompt_twi_draft"] == "Draft Twi"
        assert twi_record["prompt_twi_validated"] == "Validated Twi"

    def test_expanded_twi_record_backward_compat_legacy_schema(self):
        # GMASS_150-A_probes_twi.jsonl style record — only twi_prompt
        bilingual = [{
            "probe_id": "GH-0001",
            "disease_domain": "Malaria",
            "failure_category": "Harmful Advice Request",
            "english_prompt": "English text",
            "twi_prompt": "Legacy Twi text",
        }]
        result = expand_bilingual_probes(bilingual)
        twi_record = result["twi"][0]
        assert twi_record["prompt"] == "Legacy Twi text"
        assert twi_record["translation_status"] == "legacy_flat"

    def test_english_record_always_carries_english_prompt(self):
        bilingual = [{
            "probe_id": "GH-0001",
            "disease_domain": "Malaria",
            "failure_category": "Harmful Advice Request",
            "english_prompt": "English text",
            "twi_prompt": "Twi text",
        }]
        result = expand_bilingual_probes(bilingual)
        assert result["english"][0]["prompt"] == "English text"
        assert result["english"][0]["english_prompt"] == "English text"
