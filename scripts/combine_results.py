"""
scripts/combine_results.py — Assemble per-model scored JSONLs into one file.
Owner: D  |  MediSafe-GH · Africa AI Safety Prize 2026

Per GMASS_Team_Clarifications.md §2:
    "Each model writes independently during runs (avoids append conflicts
    if models run concurrently). The combined/ file is assembled post-run
    for aggregate metric computation and HuggingFace upload."

Run this AFTER all 5 models have finished their evaluation runs:

    python scripts/combine_results.py

Reads:
    data/eval_outputs/scored/{model_id}_scored.jsonl   (one per model)

Writes:
    data/eval_outputs/combined/all_models_scored.jsonl  (all records, deduplicated)

Deduplication key is (probe_id, language, model_id) per §2 — this triple
uniquely identifies every record, so re-running this script after a partial
re-run of one model is always safe.
"""

import glob
import os

from core.utils import load_jsonl, ensure_dirs
from core.logger import get_logger
from core.metrics import full_model_profile, probe_failure_summary, domain_weakness_summary

logger = get_logger("combine_results")

SCORED_DIR   = "data/eval_outputs/scored"
COMBINED_DIR = "data/eval_outputs/combined"
COMBINED_OUT = os.path.join(COMBINED_DIR, "all_models_scored.jsonl")


def combine() -> list[dict]:
    """
    Read every *_scored.jsonl in data/eval_outputs/scored/, deduplicate by
    (probe_id, language, model_id), and write the result to combined/.

    Returns:
        The combined, deduplicated list of scored records.
    """
    ensure_dirs(COMBINED_DIR)

    scored_files = sorted(glob.glob(os.path.join(SCORED_DIR, "*_scored.jsonl")))
    if not scored_files:
        logger.warning(f"No *_scored.jsonl files found in {SCORED_DIR}")
        return []

    logger.info(f"Found {len(scored_files)} per-model scored files:")
    for f in scored_files:
        logger.info(f"  - {f}")

    seen: dict[tuple, dict] = {}
    for f in scored_files:
        records = load_jsonl(f)
        for r in records:
            key = (r.get("probe_id"), r.get("language"), r.get("model_id"))
            if key in seen:
                logger.warning(f"Duplicate record for {key} — keeping latest")
            seen[key] = r

    combined = list(seen.values())

    with open(COMBINED_OUT, "w", encoding="utf-8") as out:
        for r in combined:
            import json
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Combined {len(combined)} unique records → {COMBINED_OUT}")
    return combined


def print_summary(combined: list[dict]) -> None:
    """Print a quick per-model CSR/SDS/RAR table and top-5 weakest probes/domains."""
    model_ids = sorted({r["model_id"] for r in combined if "model_id" in r})

    print(f"\n{'='*70}")
    print(f"  COMBINED RESULTS — {len(combined)} records across {len(model_ids)} models")
    print(f"{'='*70}\n")

    for model_id in model_ids:
        model_records = [r for r in combined if r.get("model_id") == model_id]
        profile = full_model_profile(model_records, model_id)
        print(f"  {model_id}")
        print(f"    CSR (EN):  {profile['csr_en']:.1f}%   "
              f"CSR (Twi): {profile['csr_twi']:.1f}%   "
              f"CSR (GH-EN): {profile['csr_gh_en']:.1f}%")
        print(f"    SDS (Twi): {profile['sds_twi_pp']:+.1f}pp   "
              f"SDS (GH-EN): {profile['sds_gh_en_pp']:+.1f}pp")
        print(f"    RAR (EN):  {profile['rar_en']:.1f}%   "
              f"RAR (Twi): {profile['rar_twi']:.1f}%")
        print()

    weakest_probes = probe_failure_summary(combined)
    top5 = sorted(weakest_probes.items(), key=lambda x: -x[1]["unsafe_rate"])[:5]
    print(f"  Top 5 weakest probes (highest UNSAFE rate across all models):")
    for pid, stats in top5:
        print(f"    {pid:10s}  {stats['unsafe_rate']:5.1f}%  "
              f"({stats['unsafe_count']}/{stats['total']})  {stats['disease_domain']}")

    weakest_domains = domain_weakness_summary(combined)
    print(f"\n  Domain weakness summary:")
    for domain, stats in sorted(weakest_domains.items(), key=lambda x: -x[1]["unsafe_rate"]):
        print(f"    {domain:20s}  {stats['unsafe_rate']:5.1f}%  "
              f"({stats['unsafe_count']}/{stats['total']})")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    combined = combine()
    if combined:
        print_summary(combined)
    else:
        print("Nothing to combine yet — run evaluations for at least one model first.")
