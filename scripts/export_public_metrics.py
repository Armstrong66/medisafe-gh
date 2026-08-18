"""
scripts/export_public_metrics.py — Auto-parse and export public benchmark metrics.
Owner: MediSafe-GH Team · Africa AI Safety Prize 2026

Generates public-safe metric summaries (CSR, SDS, RAR, domain breakdowns,
deploy status) from scored JSONL outputs.

Per Section 14 (Dataset Access Tiers):
    - Public metrics: aggregate percentages and readiness signals (OPEN)
    - Raw probe / model text outputs: kept separate and not exposed in public metrics

Outputs:
    data/public_metrics/benchmark_summary.json  (for Gradio dashboard & API consumers)
    data/public_metrics/benchmark_summary.md    (for README / HF Space documentation)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.metrics import (
    csr_by_domain_and_language,
    domain_weakness_summary,
    full_model_profile,
    probe_failure_summary,
)
from core.utils import ensure_dirs, load_jsonl, utc_now

logger = get_logger("export_public_metrics")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORED_DIR = ROOT / "data" / "eval_outputs" / "scored"
DEFAULT_COMBINED_FILE = ROOT / "data" / "eval_outputs" / "combined" / "all_models_scored.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "public_metrics"


def collect_scored_records(
    combined_file: Path = DEFAULT_COMBINED_FILE,
    scored_dir: Path = DEFAULT_SCORED_DIR,
) -> list[dict]:
    """Load scored records from combined JSONL or individual per-model scored files."""
    if combined_file.exists():
        records = load_jsonl(str(combined_file), warn_missing=False)
        if records:
            logger.info(f"Loaded {len(records)} records from combined: {combined_file}")
            return records

    # Fallback to loading all *_scored.jsonl in scored_dir
    records = []
    if scored_dir.exists():
        for path in sorted(scored_dir.glob("*_scored.jsonl")):
            loaded = load_jsonl(str(path), warn_missing=False)
            logger.info(f"Loaded {len(loaded)} records from {path.name}")
            records.extend(loaded)
    return records


def generate_public_metrics(scored_records: list[dict], version: str = "1.1.0") -> dict:
    """
    Compute aggregate benchmark metrics stripped of any raw prompt or response text.
    """
    model_ids = sorted({r.get("model_id") for r in scored_records if r.get("model_id")})
    profiles: dict[str, dict] = {}
    domain_breakdowns: dict[str, dict] = {}

    for model_id in model_ids:
        model_rows = [r for r in scored_records if r.get("model_id") == model_id]
        profiles[model_id] = full_model_profile(model_rows, model_id)
        domain_breakdowns[model_id] = csr_by_domain_and_language(model_rows, model_id)

    probe_summary = probe_failure_summary(scored_records)
    weakest_probes = [
        {"probe_id": pid, **stats}
        for pid, stats in sorted(probe_summary.items(), key=lambda x: -x[1]["unsafe_rate"])[:10]
    ]
    domain_summary = domain_weakness_summary(scored_records)

    payload = {
        "benchmark": "G-MASS (Ghana Medical AI Safety Screen)",
        "version": version,
        "exported_at": utc_now(),
        "total_scored_records": len(scored_records),
        "evaluated_models": model_ids,
        "profiles": profiles,
        "csr_by_domain_and_language": domain_breakdowns,
        "weakest_probes_top10": weakest_probes,
        "domain_weakness_summary": domain_summary,
    }
    return payload


def generate_markdown_summary(payload: dict) -> str:
    """Format the public metrics dictionary as a GitHub/HuggingFace-ready Markdown table."""
    profiles = payload.get("profiles", {})
    exported_at = payload.get("exported_at", "")
    version = payload.get("version", "")
    total = payload.get("total_scored_records", 0)

    lines = [
        f"## G-MASS Benchmark Results (v{version})",
        "",
        f"*Exported: {exported_at} | Total evaluated probes: {total}*",
        "",
        "### Model Safety Profiles",
        "",
        "| Model | CSR English (%) | CSR Twi (%) | CSR GH-EN (%) | SDS Twi (pp) | RAR English (%) | RAR Twi (%) | Deploy Status |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for model_id, prof in profiles.items():
        csr_en = f"{prof.get('csr_en'):.1f}" if prof.get("csr_en") is not None else "n/a"
        csr_twi = f"{prof.get('csr_twi'):.1f}" if prof.get("csr_twi") is not None else "n/a"
        csr_gh = f"{prof.get('csr_gh_en'):.1f}" if prof.get("csr_gh_en") is not None else "n/a"
        sds_twi = f"{prof.get('sds_twi_pp'):+.1f}" if prof.get("sds_twi_pp") is not None else "n/a"
        rar_en = f"{prof.get('rar_en'):.1f}" if prof.get("rar_en") is not None else "n/a"
        rar_twi = f"{prof.get('rar_twi'):.1f}" if prof.get("rar_twi") is not None else "n/a"
        status = prof.get("deploy_status", "not_evaluable")
        lines.append(
            f"| `{model_id}` | {csr_en}% | {csr_twi}% | {csr_gh}% | {sds_twi}pp | {rar_en}% | {rar_twi}% | `{status}` |"
        )

    lines.extend(
        [
            "",
            "> **Note**: CSR (Clinical Safety Rate), SDS (Safety Degradation Score), and RAR (Referral Adequacy Rate) "
            "are preliminary evaluation metrics on the G-MASS v1.0 benchmark, not clinical deployment certifications.",
            "",
        ]
    )
    return "\n".join(lines)


def export_public_metrics(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    combined_file: Path = DEFAULT_COMBINED_FILE,
    version: str = "1.1.0",
) -> tuple[Path, Path]:
    """Execute collection and export of public metrics artifacts."""
    ensure_dirs(str(output_dir))
    records = collect_scored_records(combined_file=combined_file)
    if not records:
        logger.warning("No scored records found to export public metrics from.")

    payload = generate_public_metrics(records, version=version)

    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported public metrics JSON -> {json_path}")

    md_path = output_dir / "benchmark_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_summary(payload))
    logger.info(f"Exported public metrics Markdown -> {md_path}")

    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export public-safe G-MASS benchmark metrics.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save public metric artifacts (default: data/public_metrics)",
    )
    parser.add_argument(
        "--combined-file",
        default=str(DEFAULT_COMBINED_FILE),
        help="Path to combined scored JSONL (default: data/eval_outputs/combined/all_models_scored.jsonl)",
    )
    parser.add_argument(
        "--version",
        default="1.1.0",
        help="G-MASS benchmark software version (default: 1.1.0)",
    )
    args = parser.parse_args()

    json_out, md_out = export_public_metrics(
        output_dir=Path(args.output_dir),
        combined_file=Path(args.combined_file),
        version=args.version,
    )
    print(f"\nPublic metrics successfully exported:")
    print(f"  - JSON:     {json_out}")
    print(f"  - Markdown: {md_out}")


if __name__ == "__main__":
    main()
