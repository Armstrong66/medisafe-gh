from types import SimpleNamespace

import run_bilingual_eval


def _args(skip_report=True):
    return SimpleNamespace(
        probe_file="data/probes/example.jsonl",
        delay=0,
        full=False,
        per_domain=1,
        skip_report=skip_report,
    )


def test_all_model_runner_continues_after_model_failure(monkeypatch):
    calls = []

    def fake_run(cmd, check):
        assert check is False
        model_key = cmd[2]
        calls.append(model_key)
        return SimpleNamespace(returncode=7 if model_key == "gemini" else 0)

    monkeypatch.setattr(run_bilingual_eval.subprocess, "run", fake_run)

    result = run_bilingual_eval.run_all_models_and_report(_args(skip_report=True))

    assert result == 1
    assert calls == run_bilingual_eval.PROBE_TESTED_MODEL_KEYS


def test_all_model_runner_attempts_report_after_partial_failure(monkeypatch):
    calls = []
    report_calls = []

    def fake_run(cmd, check):
        model_key = cmd[2]
        calls.append(model_key)
        return SimpleNamespace(returncode=1 if model_key == "gpt4o" else 0)

    def fake_combine():
        report_calls.append("combine")
        return [{"model_id": "gemini"}]

    def fake_print_summary(rows):
        report_calls.append(("summary", len(rows)))

    def fake_build_report(combined_out, report_path):
        report_calls.append(("report", combined_out, report_path))

    monkeypatch.setattr(run_bilingual_eval.subprocess, "run", fake_run)
    monkeypatch.setattr("scripts.combine_results.combine", fake_combine)
    monkeypatch.setattr("scripts.combine_results.print_summary", fake_print_summary)
    monkeypatch.setattr("scripts.build_evaluation_report.build_report", fake_build_report)

    result = run_bilingual_eval.run_all_models_and_report(_args(skip_report=False))

    assert result == 1
    assert calls == run_bilingual_eval.PROBE_TESTED_MODEL_KEYS
    assert report_calls[0] == "combine"
    assert report_calls[1] == ("summary", 1)
    assert report_calls[2][0] == "report"
