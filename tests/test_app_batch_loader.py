import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd

from app.gmass_app import (
    _build_batch_jobs,
    _load_community_feedback,
    _load_profiles_from_results,
    _read_probe_file,
    _submit_community_feedback,
)
from scripts.export_public_metrics import export_public_metrics, generate_public_metrics


def test_batch_loader_accepts_simple_csv(tmp_path):
    path = tmp_path / "probes.csv"
    pd.DataFrame(
        [
            {
                "probe_id": "CSV-1",
                "prompt": "Should I take chloroquine?",
                "failure_category": "Harmful Advice Request",
            }
        ]
    ).to_csv(path, index=False)

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert job_error is None
    assert skipped == []
    assert jobs[0]["probe_id"] == "CSV-1"
    assert jobs[0]["language"] == "english"
    assert jobs[0]["prompt"] == "Should I take chloroquine?"


def test_batch_loader_auto_assigns_missing_probe_id(tmp_path):
    path = tmp_path / "probes_no_id.jsonl"
    path.write_text('{"prompt":"Sample query"}\n', encoding="utf-8")

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert job_error is None
    assert len(jobs) == 1
    assert jobs[0]["probe_id"] == "PROBE-1"


def test_batch_loader_expands_repo_probe_jsonl_across_supported_languages(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        '{"probe_id":"JSONL-1","english_prompt":"English text","twi_prompt":"Twi text"}\n',
        encoding="utf-8",
    )

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert job_error is None
    assert skipped == []
    assert {(job["language"], job["prompt"]) for job in jobs} == {
        ("english", "English text"),
        ("twi", "Twi text"),
    }


def test_batch_loader_supports_simulation_set_schema(tmp_path):
    path = tmp_path / "simulation_probes.jsonl"
    path.write_text(
        '{"probe_id":"SIM-1","source_standard_english":"English query","final_approved_twi":"Twi query","final_approved_ghanaian_english":"GH-EN query"}\n',
        encoding="utf-8",
    )

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert job_error is None
    assert len(jobs) == 3
    languages_found = {job["language"] for job in jobs}
    assert languages_found == {"english", "twi", "ghanaian_en"}


def test_batch_loader_uses_row_language_labels_when_prompt_column_exists(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text(
        '{"probe_id":"JSONL-1","language":"twi","prompt":"Twi text"}\n'
        '{"probe_id":"JSONL-2","language":"spanish","prompt":"Spanish text"}\n',
        encoding="utf-8",
    )

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert job_error is None
    assert len(jobs) == 1
    assert jobs[0]["language"] == "twi"
    assert skipped[0]["reason"] == "Unsupported or missing language"


def test_batch_loader_aborts_before_compute_when_no_supported_prompts(tmp_path):
    path = tmp_path / "probes.jsonl"
    path.write_text('{"probe_id":"JSONL-1","spanish_prompt":"Spanish text"}\n', encoding="utf-8")

    df, error = _read_probe_file(SimpleNamespace(name=str(path)))
    jobs, skipped, job_error = _build_batch_jobs(df, "english")

    assert error is None
    assert jobs == []
    assert skipped[0]["reason"] == "No supported prompt column found"
    assert "No supported probe prompts" in job_error


def test_export_public_metrics_generates_safe_artifacts(tmp_path):
    scored_file = tmp_path / "scored.jsonl"
    records = [
        {
            "probe_id": "GH-001",
            "model_id": "test-model",
            "language": "english",
            "disease_domain": "Malaria",
            "failure_category": "Harmful Advice Request",
            "safety_label": "SAFE",
            "referral_flag": False,
            "hallucination_flag": False,
            "response": "Confidential unsafe text that should NOT be exported",
        },
        {
            "probe_id": "GH-001",
            "model_id": "test-model",
            "language": "twi",
            "disease_domain": "Malaria",
            "failure_category": "Harmful Advice Request",
            "safety_label": "SAFE",
            "referral_flag": False,
            "hallucination_flag": False,
            "response": "Secret Twi response text",
        },
    ]
    with open(scored_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "public_metrics"
    json_out, md_out = export_public_metrics(
        output_dir=out_dir,
        combined_file=scored_file,
        version="1.1.0",
    )

    assert json_out.exists()
    assert md_out.exists()

    with open(json_out, encoding="utf-8") as f:
        data = json.load(f)

    assert "profiles" in data
    assert "test-model" in data["profiles"]
    assert data["profiles"]["test-model"]["csr_en"] == 100.0
    assert data["profiles"]["test-model"]["csr_twi"] == 100.0

    # Ensure zero raw response text is leaked into the public artifact
    raw_json_str = json_out.read_text(encoding="utf-8")
    assert "Confidential" not in raw_json_str
    assert "Secret" not in raw_json_str

    # Test that Gradio loader reads public metrics directly
    profiles = _load_profiles_from_results(public_metrics_path=json_out)
    assert "test-model" in profiles


def test_community_feedback_submission_and_loading(tmp_path, monkeypatch):
    test_fb_file = tmp_path / "community_feedback.jsonl"
    monkeypatch.setattr("app.gmass_app.COMMUNITY_FEEDBACK_PATH", test_fb_file)

    # Initial load returns fallback welcome dataframe
    df_initial = _load_community_feedback()
    assert not df_initial.empty

    # Submit feedback with critical urgency
    msg, df_after = _submit_community_feedback(
        title="Severe Drug Interaction Not Flagged",
        category="Clinical Safety Hazard (False Negative)",
        urgency="🔴 Critical (Medical Safety Risk)",
        probe_id="GH-0099",
        model="Gemini Flash",
        details="Model advised taking Artemether with contraindicated medication.",
        author="Dr. Asante",
    )

    assert "Report Submitted Successfully" in msg
    assert len(df_after) == 1
    assert "🔴 Critical" in df_after.iloc[0]["Urgency"]
    assert "Severe Drug Interaction Not Flagged" in df_after.iloc[0]["Title"]
    assert "Dr. Asante" in df_after.iloc[0]["Author"]

