# run_bilingual_eval.py
# MediSafe-GH · G-MASS Project
# Team D — Engineering Lead
#
# Runs the same probe set through a model in BOTH English and Twi,
# scores both, and computes SDS (Safety Degradation Score) immediately.
#
# REVISED per GMASS_Team_Clarifications.md:
#   §2 — output files are one-JSONL-per-model (data/eval_outputs/raw/<model>.jsonl,
#        data/eval_outputs/scored/<model>_scored.jsonl), NOT one file per
#        eval-run-type. All language conditions for a model accumulate into
#        the SAME file, distinguished by the language field on each record.
#   §8 — Twi/GH-EN prompts get the language-consistency instruction appended
#        before being sent to the model (build_prompt_with_language_instruction).
#   §9 — GPT-4o mini, not full GPT-4o (model ID map updated).
#
# Usage:
#   python run_bilingual_eval.py gemini
#   python run_bilingual_eval.py gemini --per-domain 5    (pilot mode)
#   python run_bilingual_eval.py gemini --full            (all 150 probes)
#   python run_bilingual_eval.py all --per-domain 5       (run all probe-tested models + report)
#   python run_bilingual_eval.py all --full               (full suite + report)

import argparse
import os
import subprocess
import sys
import time

from probes.loader import load_bilingual_probes, expand_bilingual_probes
from scorer.scorer import gmass_score
from core.utils import save_jsonl_line, load_jsonl, ensure_dirs
from core.metrics import full_model_profile
from core.logger import get_logger
from models.router import call_model, build_prompt_with_language_instruction

logger = get_logger("run_bilingual_eval")

# Model IDs are read directly from models/router.py's own constants rather
# than duplicated here — this exact duplication (a hardcoded copy drifting
# out of sync with router.py's real defaults) is what caused this script to
# still say "gpt-4o-mini" and "gemini-2.5-flash" after the team decided to
# reinstate the probe-tested model lineup. Importing the live values means
# this map can never silently go stale again.
from models.router import GPT4O_MODEL, GEMINI_MODEL, PHI3_MODEL, BIOMISTRAL_MODEL

MODEL_ID_MAP = {
    "gpt4o":      GPT4O_MODEL,
    "gemini":     GEMINI_MODEL,
    "phi3":       PHI3_MODEL,
    "biomistral": BIOMISTRAL_MODEL,
}
PROBE_TESTED_MODEL_KEYS = list(MODEL_ID_MAP.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bilingual G-MASS evaluation")
    parser.add_argument("model", help="Model key: gpt4o, gemini, phi3, biomistral, or all")
    parser.add_argument(
        "--probe-file",
        default="data/probes/probes_bilingual.jsonl",
        help="Path to bilingual/GH-EN probe JSONL (default: data/probes/probes_bilingual.jsonl)",
    )
    parser.add_argument(
        "--per-domain",
        type=int,
        default=5,
        help="Probes per domain for pilot mode (default: 5)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 150 probes instead of pilot sample",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between API calls (default: 2.0)",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="With model=all, run evaluations only; do not combine outputs or build the workbook",
    )
    return parser.parse_args(argv)


def run_all_models_and_report(args: argparse.Namespace) -> int:
    """
    Run every probe-tested model, then combine results and build the report.

    One provider's quota, token, API, or local backend failure must not stop
    the remaining available models. Each model runs in its own subprocess;
    failures are recorded, and the lineup continues.
    """
    failed_models: list[tuple[str, int]] = []

    for model_key in PROBE_TESTED_MODEL_KEYS:
        cmd = [
            sys.executable, str(os.path.abspath(__file__)), model_key,
            "--probe-file", args.probe_file,
            "--delay", str(args.delay),
        ]
        if args.full:
            cmd.append("--full")
        else:
            cmd.extend(["--per-domain", str(args.per_domain)])
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failed_models.append((model_key, result.returncode))
            print(
                f"\nWARNING: {model_key} run failed with exit code {result.returncode}. "
                "Continuing with remaining models."
            )

    if args.skip_report:
        print("\nAll probe-tested model runs attempted. Report generation skipped.")
        _print_all_model_failures(failed_models)
        return 1 if failed_models else 0

    from scripts.combine_results import COMBINED_OUT, combine, print_summary
    from scripts.build_evaluation_report import build_report

    combined = combine()
    if combined:
        print_summary(combined)

    report_path = "data/eval_outputs/combined/GMASS_Evaluation_Results.xlsx"
    build_report(COMBINED_OUT, report_path)

    print("\nFull evaluation pipeline complete.")
    print(f"Combined results: {COMBINED_OUT}")
    print(f"Workbook report:   {report_path}")
    _print_all_model_failures(failed_models)
    return 1 if failed_models else 0


def _print_all_model_failures(failed_models: list[tuple[str, int]]) -> None:
    if not failed_models:
        return
    print("\nPartial run warning: these model runs failed, likely due to provider/API/local issues:")
    for model_key, returncode in failed_models:
        print(f"  - {model_key}: exit code {returncode}")
    print("Other model outputs were still preserved and report generation was attempted.")


def normalize_probe_schema(probes: list[dict]) -> list[dict]:
    """Support both canonical probes and approved simulation-set field names."""
    normalized = []
    for probe in probes:
        p = dict(probe)
        if "english_prompt" not in p and "source_standard_english" in p:
            p["english_prompt"] = p["source_standard_english"]
        if "prompt_twi_validated" not in p and "final_approved_twi" in p:
            p["prompt_twi_validated"] = p["final_approved_twi"]
        if "twi_prompt" not in p and "final_approved_twi" in p:
            p["twi_prompt"] = p["final_approved_twi"]
        if "ghanaian_en_prompt" not in p and "final_approved_ghanaian_english" in p:
            p["ghanaian_en_prompt"] = p["final_approved_ghanaian_english"]
        missing = [
            field for field in ("probe_id", "disease_domain", "failure_category", "english_prompt")
            if field not in p
        ]
        if missing:
            raise KeyError(f"Probe {p.get('probe_id', '<unknown>')} missing required fields: {missing}")
        normalized.append(p)
    return normalized


def run_language(
    language: str,
    probes: list[dict],
    model_key: str,
    model_id: str,
    raw_out: str,
    scored_out: str,
    delay: float,
):
    """Run one language condition through the model and score it."""
    completed_lang_keys = set()

    # Need probe_id + language as the unique key since both languages
    # share the same probe_id, and this file accumulates across runs (§2)
    if os.path.exists(scored_out):
        existing = load_jsonl(scored_out, warn_missing=False)
        completed_lang_keys = {(r["probe_id"], r["language"]) for r in existing}

    pending = [p for p in probes if (p["probe_id"], language) not in completed_lang_keys]

    print(f"\n{'─'*60}")
    print(f"  Language: {language.upper()}  |  Model: {model_key.upper()} ({model_id})")
    print(f"  Pending: {len(pending)} probes  "
          f"(skipping {len(probes) - len(pending)} already done)")
    print(f"{'─'*60}\n")

    for i, probe in enumerate(pending, 1):
        pid       = probe["probe_id"]
        prompt    = probe["prompt"]           # language-specific — sent TO the model
        prompt_en = probe["english_prompt"]   # always English — sent to the SCORER
        category  = probe["failure_category"]

        # §8: append "respond in the same language" instruction for Twi/GH-EN.
        # No-op for English. Non-compliance is checked downstream by
        # LanguageConsistencyChecker in scorer/scorer.py and
        # surfaced as language_consistency_ok in the scored output.
        prompt_to_send = build_prompt_with_language_instruction(prompt, language)

        try:
            t0       = time.time()
            response = call_model(model_key, prompt_to_send)
            latency  = int((time.time() - t0) * 1000)

            # §1: carry translation_status (validated/draft_unreviewed/legacy_flat)
            # into the raw output so the audit trail of which Twi text was
            # actually used survives into the dataset, without needing to
            # re-join against the source probe file later.
            raw_record = {
                "probe_id":   pid,
                "model_id":   model_id,
                "language":   language,
                "response":   response,
                "latency_ms": latency,
            }
            if language == "twi":
                raw_record["translation_status"] = probe.get("translation_status", "unknown")
            save_jsonl_line(raw_record, raw_out)

            # prompt_en (always English) is passed to the scorer, not prompt.
            # gmass_score() needs the ENGLISH probe for LlamaGuard3 context
            # regardless of what language the model was actually queried in.
            # The response itself is routed by gmass_score() to AfroLM
            # (native, primary for Twi) and/or LlamaGuard3 (via Khaya
            # back-translation, secondary for Twi) per clarifications §7.
            scored = gmass_score(pid, model_id, language, category, prompt_en, response, latency)
            save_jsonl_line(scored, scored_out)

            icon = "✓" if scored["safety_label"] == "SAFE" else "✗"
            lang_flag = ""
            if scored.get("language_consistency_ok") is False:
                lang_flag = "  ⚠ responded in wrong language"
            print(f"  [{i:>3}/{len(pending)}] {pid:10s} → {scored['safety_label']:6s}  {icon}{lang_flag}")

        except Exception as e:
            logger.error(f"[{pid}] [{language}] Failed: {e}")
            print(f"  [{i:>3}/{len(pending)}] {pid:10s} → ERROR: {str(e)[:60]}")

        if i < len(pending):
            time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model_key = args.model

    if model_key == "all":
        return run_all_models_and_report(args)

    if model_key not in MODEL_ID_MAP:
        raise SystemExit(
            f"Unknown model: '{model_key}'. Valid options: {list(MODEL_ID_MAP.keys()) + ['all']}"
        )

    model_id = MODEL_ID_MAP[model_key]

    ensure_dirs("data/eval_outputs/raw", "data/eval_outputs/scored", "logs")

    # ── Load and expand bilingual probes ───────────────────────────────────────
    bilingual = normalize_probe_schema(load_bilingual_probes(args.probe_file))
    expanded = expand_bilingual_probes(bilingual)

    # §11: GH-EN was always in scope (clerical correction, not new scope) and does
    # NOT depend on Twi validator review — start it now, in parallel. If the probe
    # file has a ghanaian_en_prompt field, expand it the same way as English/Twi.
    # GH-EN is always evaluated as the third language condition. If the probe
    # file provides a dedicated ghanaian_en_prompt field, use it. Otherwise, use
    # the English prompt with the Ghanaian-English response instruction appended
    # downstream by build_prompt_with_language_instruction().
    gh_en_has_dedicated_prompt = bool(bilingual) and "ghanaian_en_prompt" in bilingual[0]
    expanded["ghanaian_en"] = [
        {
            "probe_id":         p["probe_id"],
            "disease_domain":   p["disease_domain"],
            "failure_category": p["failure_category"],
            "english_prompt":   p["english_prompt"],
            "language":         "ghanaian_en",
            "prompt":           p.get("ghanaian_en_prompt") or p["english_prompt"],
        }
        for p in bilingual
    ]
    if gh_en_has_dedicated_prompt:
        logger.info(f"GH-EN dedicated prompts available - {len(expanded['ghanaian_en'])} records")
    else:
        logger.info(
            "No ghanaian_en_prompt field in probe file - using english_prompt plus "
            "the Ghanaian-English response instruction for GH-EN condition."
        )

    if not args.full:
        # Pilot mode — take N per domain from each language, matched by probe_id
        domain_counts = {}
        pilot_ids = set()
        for p in expanded["english"]:
            d = p["disease_domain"]
            if domain_counts.get(d, 0) < args.per_domain:
                pilot_ids.add(p["probe_id"])
                domain_counts[d] = domain_counts.get(d, 0) + 1

        for lang in ("english", "twi", "ghanaian_en"):
            expanded[lang] = [p for p in expanded[lang] if p["probe_id"] in pilot_ids]

        total_calls = sum(len(expanded[lang]) for lang in ("english", "twi", "ghanaian_en"))
        print(f"\nPILOT MODE: {len(pilot_ids)} probe_ids × "
              f"3 languages = {total_calls} total calls")
    else:
        total_calls = sum(len(expanded[lang]) for lang in ("english", "twi", "ghanaian_en"))
        print(f"\nFULL MODE: {len(expanded['english'])} probe_ids × "
              f"3 languages = {total_calls} total calls")

    # §2: ONE file per model — not one per eval-run-type. All language conditions
    # for this model accumulate into the same raw/scored JSONL, distinguished by
    # the "language" field on each record. probe_id + language + model_id is the
    # unique key (no duplication possible across separate runs of this script).
    raw_out = f"data/eval_outputs/raw/{model_id}.jsonl"
    scored_out = f"data/eval_outputs/scored/{model_id}_scored.jsonl"

    print(f"\nStarting bilingual evaluation: {model_key.upper()} ({model_id})")
    print(f"Raw output:    {raw_out}")
    print(f"Scored output: {scored_out}")

    run_language("english", expanded["english"], model_key, model_id, raw_out, scored_out, args.delay)
    run_language("twi", expanded["twi"], model_key, model_id, raw_out, scored_out, args.delay)
    run_language("ghanaian_en", expanded["ghanaian_en"], model_key, model_id, raw_out, scored_out, args.delay)

    # ── Compute SDS ─────────────────────────────────────────────────────────────
    scored_outputs = load_jsonl(scored_out)
    profile = full_model_profile(scored_outputs, model_id)

    print(f"\n\n{'='*60}")
    print(f"  RESULTS — {model_id}")
    print(f"{'='*60}")
    print(f"  CSR (English): {profile['csr_en']}%")
    print(f"  CSR (Twi):     {profile['csr_twi']}%")
    print(f"  CSR (GH-EN):   {profile['csr_gh_en']}%")
    print(f"  RAR (English): {profile['rar_en']}%")
    print(f"  RAR (Twi):     {profile['rar_twi']}%")
    print(f"\n  Safety Degradation Score — Twi:   {profile['sds_twi_pp']:+.1f}pp")
    print(f"  Safety Degradation Score — GH-EN: {profile['sds_gh_en_pp']:+.1f}pp")
    print(f"  Deploy status: {profile['deploy_status']}")

    if profile["deploy_status"] == "ready":
        print("  ✓ Meets the configured readiness gates")
    elif profile["deploy_status"] == "not_ready":
        print("  ⚠ Fails one or more readiness gates")
    else:
        print("  • Not evaluable: insufficient denominator rows for one or more required checks")

    # §13: explicit reminder against overclaiming
    print("\n  NOTE: Do not report this as 'Model is safe for Ghanaian medical use.'")
    print(f"  Report as: 'Model showed an SDS of {profile['sds_twi_pp']:+.1f}pp on this")
    print("  v1.0 benchmark — a preliminary signal, not a deployment certification.'")
    print(f"{'='*60}\n")

    print(f"Scored results saved to: {scored_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
