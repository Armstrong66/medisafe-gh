"""
medisafe_gh.cli — G-MASS command-line interface.

Entry point for the `gmass` command installed via:
    pip install medisafe-gh
    docker run medisafe-gh:cpu gmass <command>

Commands
--------
gmass evaluate   Run G-MASS safety evaluation on one or all models
gmass score      Score already-collected raw model outputs
gmass report     Generate the Clinical Simulation Report from scored outputs
gmass profile    Print the full G-MASS safety profile for a model
gmass probe      Utilities for inspecting and validating the probe set
gmass --version  Print version and exit

Design notes
------------
- All commands are self-contained: they load config, set up logging,
  and run cleanly in interactive, terminal, or nohup/Docker contexts.
- API keys are always loaded from .env or environment — never from CLI args.
- Every command respects GMASS_LOG_LEVEL for verbosity control.
- Long-running commands (evaluate) log structured headers/footers so
  nohup log files are parseable and identifiable after the fact.

Owner: A (Team Lead)
"""

import argparse
import sys
import time
from pathlib import Path

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "0.1.0"

# ── Lazy imports (keep CLI startup fast; heavy deps load only when needed) ────
def _import_core():
    from medisafe_gh.core.logger import get_logger, log_run_header, log_run_footer
    from medisafe_gh.core.config import load_config
    from medisafe_gh.core.utils  import (
        load_jsonl, append_jsonl, load_completed_ids,
        log_environment, is_kaggle, is_cuda_available,
    )
    return get_logger, log_run_header, log_run_footer, load_config, \
           load_jsonl, append_jsonl, load_completed_ids, log_environment


# ── Shared paths ──────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
CONFIG_DIR  = REPO_ROOT / "configs"
DATA_DIR    = REPO_ROOT / "data"
PROBES_DIR  = DATA_DIR  / "probes"
RAW_DIR     = DATA_DIR  / "eval_outputs" / "raw"
SCORED_DIR  = DATA_DIR  / "eval_outputs" / "scored"
COMBINED    = DATA_DIR  / "eval_outputs" / "combined" / "all_models_scored.jsonl"
SIM_DIR     = DATA_DIR  / "simulation"


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: evaluate
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_evaluate(args):
    """
    Run G-MASS safety evaluation: load probes → call model API/local
    inference → append raw responses → score → append scored outputs.

    Supports crash-safe resumption: already-completed probe IDs are
    skipped automatically via load_completed_ids().
    """
    from dotenv import load_dotenv
    load_dotenv()

    get_logger, log_run_header, log_run_footer, load_config, \
    load_jsonl, append_jsonl, load_completed_ids, log_environment = _import_core()

    logger = get_logger("gmass.evaluate")
    log_environment(logger)

    # Load configs
    gmass_cfg  = load_config("gmass_config")
    models_cfg = load_config("models")

    # Resolve which models to run
    all_models = models_cfg["models"]
    if args.model and args.model != "all":
        target_models = [m for m in all_models if m["id"] == args.model]
        if not target_models:
            logger.error(f"Model '{args.model}' not found in configs/models.yaml")
            sys.exit(1)
    else:
        target_models = all_models

    # Resolve which languages to run
    all_langs = gmass_cfg["languages"]
    if args.language and args.language != "all":
        if args.language not in all_langs:
            logger.error(f"Language '{args.language}' not in gmass_config.yaml languages list")
            sys.exit(1)
        target_langs = [args.language]
    else:
        target_langs = all_langs

    # Load probe set
    probe_file = PROBES_DIR / f"probes_{args.probe_split}.jsonl"
    if not probe_file.exists():
        logger.error(
            f"Probe file not found: {probe_file}\n"
            f"Run `gmass probe check` to verify probe set integrity."
        )
        sys.exit(1)

    all_probes = load_jsonl(probe_file)
    logger.info(f"Loaded {len(all_probes)} probes from {probe_file.name}")

    # Lazy-load the scorer and model caller
    from medisafe_gh.scoring.scorer  import GMassScorer
    from medisafe_gh.core.evaluate   import call_model, build_prompt

    scorer = GMassScorer(use_cloudflare=args.cloudflare)

    total_completed = 0
    start_time      = time.time()

    for model in target_models:
        model_id   = model["id"]
        raw_path   = RAW_DIR    / f"{model_id}.jsonl"
        scored_path= SCORED_DIR / f"{model_id}_scored.jsonl"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        SCORED_DIR.mkdir(parents=True, exist_ok=True)

        for lang in target_langs:
            # Resume support: skip already-done (probe_id, language) pairs
            done_ids = load_completed_ids(raw_path)
            lang_probes = [
                p for p in all_probes
                if p.get("language") == lang
                and f"{p['probe_id']}_{lang}" not in done_ids
            ]

            if not lang_probes:
                logger.info(f"[{model_id}][{lang}] All probes already complete — skipping.")
                continue

            log_run_header(logger, {
                "model":       model_id,
                "language":    lang,
                "n_probes":    len(lang_probes),
                "raw_output":  str(raw_path),
                "scored_output": str(scored_path),
                "dry_run":     args.dry_run,
            })

            n_done = 0
            for probe in lang_probes:
                probe_id = probe["probe_id"]

                if args.dry_run:
                    logger.info(f"[DRY RUN] Would evaluate probe={probe_id} model={model_id} lang={lang}")
                    n_done += 1
                    continue

                # 1. Call model
                prompt    = build_prompt(probe, lang)
                response  = call_model(model, prompt, logger)

                if response is None:
                    logger.warning(f"Skipping probe={probe_id} — model call returned None after retries")
                    continue

                # 2. Save raw output
                raw_record = {
                    "probe_id":         probe_id,
                    "model_id":         model_id,
                    "language":         lang,
                    "failure_category": probe.get("failure_category"),
                    "disease_domain":   probe.get("disease_domain"),
                    "response":         response,
                    "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_jsonl(raw_record, raw_path)

                # 3. Score response
                result = scorer.score_one(
                    probe_id          = probe_id,
                    model_id          = model_id,
                    language          = lang,
                    failure_category  = probe.get("failure_category", ""),
                    probe_prompt_en   = probe.get("prompt_en", probe.get("prompt", "")),
                    model_response    = response,
                    model_response_en = None,  # back-translation handled inside scorer if needed
                )
                append_jsonl(result.to_dict(), scored_path)

                n_done += 1
                if n_done % 25 == 0:
                    logger.info(f"[{model_id}][{lang}] Progress: {n_done}/{len(lang_probes)}")

            total_completed += n_done
            log_run_footer(logger, n_done, len(lang_probes), time.time() - start_time)

    logger.info(f"Evaluation complete. Total records written: {total_completed}")

    if args.combine:
        logger.info("Combining all scored outputs → combined/all_models_scored.jsonl")
        cmd_combine(None)


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: score  (score raw outputs that already exist)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_score(args):
    """Score existing raw outputs without re-running model inference."""
    from dotenv import load_dotenv
    load_dotenv()

    get_logger, _, _, load_config, load_jsonl, append_jsonl, \
    load_completed_ids, log_environment = _import_core()

    logger = get_logger("gmass.score")
    log_environment(logger)

    from medisafe_gh.scoring.scorer import GMassScorer
    scorer = GMassScorer(use_cloudflare=args.cloudflare)

    raw_files = list(RAW_DIR.glob("*.jsonl")) if not args.model \
                else [RAW_DIR / f"{args.model}.jsonl"]

    for raw_path in raw_files:
        if not raw_path.exists():
            logger.warning(f"Raw file not found: {raw_path} — skipping")
            continue

        model_id    = raw_path.stem
        scored_path = SCORED_DIR / f"{model_id}_scored.jsonl"
        SCORED_DIR.mkdir(parents=True, exist_ok=True)

        raw_records = load_jsonl(raw_path)
        done_ids    = load_completed_ids(scored_path)

        pending = [r for r in raw_records
                   if f"{r['probe_id']}_{r['language']}" not in done_ids]

        logger.info(f"[{model_id}] Scoring {len(pending)}/{len(raw_records)} pending records")

        for rec in pending:
            result = scorer.score_one(
                probe_id         = rec["probe_id"],
                model_id         = rec["model_id"],
                language         = rec["language"],
                failure_category = rec.get("failure_category", ""),
                probe_prompt_en  = rec.get("probe_prompt_en", ""),
                model_response   = rec["response"],
            )
            append_jsonl(result.to_dict(), scored_path)

        logger.info(f"[{model_id}] Scoring complete → {scored_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: profile  (compute and print CSR / SDS / RAR for a model)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_profile(args):
    """Compute and print the G-MASS safety profile for one or all models."""
    get_logger, _, _, _, load_jsonl, _, _, _ = _import_core()
    from medisafe_gh.core.metrics import full_model_profile, summarise_all_models

    logger   = get_logger("gmass.profile")
    profiles = []

    scored_files = list(SCORED_DIR.glob("*_scored.jsonl")) if not args.model \
                   else [SCORED_DIR / f"{args.model}_scored.jsonl"]

    for sf in scored_files:
        if not sf.exists():
            logger.warning(f"Scored file not found: {sf} — run `gmass score` first")
            continue
        model_id = sf.stem.replace("_scored", "")
        outputs  = load_jsonl(sf)
        profile  = full_model_profile(outputs, model_id)
        profiles.append(profile)

    if not profiles:
        logger.error("No scored outputs found. Run `gmass evaluate` or `gmass score` first.")
        sys.exit(1)

    summarise_all_models(profiles)

    # Optional: write profiles to JSON for downstream report generation
    if args.save:
        import json
        out = DATA_DIR / "eval_outputs" / "model_profiles.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        logger.info(f"Profiles saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: report  (generate Clinical Simulation Report)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_report(args):
    """
    Generate the Ghana Clinical Simulation Report from scored simulation outputs.
    Reads from data/simulation/scenarios_scored.jsonl and writes a markdown report.
    """
    get_logger, _, _, _, load_jsonl, _, _, _ = _import_core()
    logger = get_logger("gmass.report")

    scored_sim = SIM_DIR / "scenarios_scored.jsonl"
    if not scored_sim.exists():
        logger.error(
            f"Scored simulation file not found: {scored_sim}\n"
            "Run `gmass evaluate --probe-split simulation` first."
        )
        sys.exit(1)

    records = load_jsonl(scored_sim)
    logger.info(f"Loaded {len(records)} scored simulation records")

    # Group by scenario then model
    from collections import defaultdict
    by_scenario = defaultdict(list)
    for r in records:
        by_scenario[r.get("scenario_id", r["probe_id"])].append(r)

    out_path = SIM_DIR / "Ghana_Clinical_Simulation_Report.md"
    lines    = [
        "# Ghana Clinical Simulation Report",
        "**G-MASS · MediSafe-GH · Africa AI Safety Prize 2026**\n",
        f"*Generated by `gmass report`*\n",
        "---\n",
        "## Summary\n",
        f"- **Scenarios evaluated:** {len(by_scenario)}",
        f"- **Total transcripts:** {len(records)}",
        "",
    ]

    for scen_id, scen_records in sorted(by_scenario.items()):
        first  = scen_records[0]
        domain = first.get("disease_domain", "")
        cat    = first.get("failure_category", "")
        prompt = first.get("prompt", first.get("probe_prompt_en", ""))

        lines += [
            f"---",
            f"## {scen_id} — {domain} / {cat}",
            f"**Prompt:** {prompt}\n",
        ]

        n_safe   = sum(1 for r in scen_records if r.get("safety_label") == "SAFE")
        n_unsafe = len(scen_records) - n_safe
        lines   += [
            f"**Results across {len(scen_records)} model runs:** "
            f"{n_safe} SAFE / {n_unsafe} UNSAFE\n",
        ]

        for rec in scen_records:
            label    = rec.get("safety_label", "—")
            referral = "✓" if rec.get("referral_flag")    else "✗"
            halluc   = "⚠" if rec.get("hallucination_flag") else "—"
            lines += [
                f"### Model: `{rec.get('model_id')}` [{rec.get('language','').upper()}]",
                f"- **Safety label:** `{label}`  "
                f"| Referral: {referral}  | Hallucination flag: {halluc}",
                f"- **Response:**\n\n> {rec.get('response', '').strip()}\n",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Simulation report written → {out_path}")
    print(f"[G-MASS] Report saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND: probe  (probe set utilities)
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_probe(args):
    """Probe set utilities: check integrity, show stats, filter."""
    get_logger, _, _, _, load_jsonl, _, _, _ = _import_core()
    logger = get_logger("gmass.probe")

    if args.probe_action == "check":
        required_fields = [
            "probe_id", "disease_domain", "failure_category",
            "prompt", "language", "validation_status",
            "expected_safe_behaviour", "unsafe_indicators",
        ]
        for split in ["en", "twi", "gh_en"]:
            path = PROBES_DIR / f"probes_{split}.jsonl"
            if not path.exists():
                logger.warning(f"Missing: {path.name}")
                continue
            probes = load_jsonl(path)
            missing_fields = []
            for p in probes:
                for f in required_fields:
                    if f not in p:
                        missing_fields.append((p.get("probe_id"), f))
            if missing_fields:
                logger.warning(f"{path.name}: {len(missing_fields)} missing fields")
                for pid, field in missing_fields[:10]:
                    logger.warning(f"  probe_id={pid} missing field: {field}")
            else:
                logger.info(f"{path.name}: {len(probes)} probes — all required fields present ✓")

    elif args.probe_action == "stats":
        from collections import Counter
        for split in ["en", "twi", "gh_en"]:
            path = PROBES_DIR / f"probes_{split}.jsonl"
            if not path.exists():
                continue
            probes = load_jsonl(path)
            domains    = Counter(p.get("disease_domain")   for p in probes)
            categories = Counter(p.get("failure_category") for p in probes)
            statuses   = Counter(p.get("validation_status") for p in probes)
            print(f"\n=== {path.name} ({len(probes)} probes) ===")
            print("By domain:   ", dict(domains))
            print("By category: ", dict(categories))
            print("By status:   ", dict(statuses))


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL: combine scored outputs
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_combine(args):
    """Combine all per-model scored JSONL files into one combined file."""
    get_logger, _, _, _, load_jsonl, append_jsonl, _, _ = _import_core()
    logger = get_logger("gmass.combine")

    COMBINED.parent.mkdir(parents=True, exist_ok=True)
    # Start fresh for the combined file
    if COMBINED.exists():
        COMBINED.unlink()

    total = 0
    for sf in sorted(SCORED_DIR.glob("*_scored.jsonl")):
        records = load_jsonl(sf)
        for r in records:
            append_jsonl(r, COMBINED)
        total += len(records)
        logger.info(f"  Added {len(records)} records from {sf.name}")

    logger.info(f"Combined file written: {COMBINED} ({total} total records)")


# ═══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmass",
        description=(
            "G-MASS — Ghana Medical AI Safety Screen\n"
            "Cross-lingual safety evaluation for medical AI in Ghanaian languages.\n"
            "MediSafe-GH · Africa AI Safety Prize 2026"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  gmass evaluate --model gpt-4o-mini --language english\n"
            "  gmass evaluate --all-models --language all --combine\n"
            "  gmass evaluate --model llama-3.2-3b --dry-run\n"
            "  gmass score --model gemini-flash\n"
            "  gmass profile\n"
            "  gmass profile --model gpt-4o-mini --save\n"
            "  gmass report\n"
            "  gmass probe check\n"
            "  gmass probe stats\n"
        ),
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"gmass {__version__}",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── evaluate ──────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Run G-MASS evaluation pipeline")
    ev.add_argument(
        "--model", default="all",
        help="Model ID from configs/models.yaml, or 'all' (default: all)",
    )
    ev.add_argument(
        "--language", default="all",
        choices=["english", "twi", "ghanaian_en", "all"],
        help="Language condition to evaluate (default: all)",
    )
    ev.add_argument(
        "--probe-split", default="en",
        choices=["en", "twi", "gh_en", "simulation"],
        help="Which probe JSONL file to use (default: en)",
    )
    ev.add_argument(
        "--cloudflare", action="store_true",
        help="Use Cloudflare Workers AI for LlamaGuard3 (Kaggle fallback)",
    )
    ev.add_argument(
        "--combine", action="store_true",
        help="After evaluation, combine all scored outputs into combined/all_models_scored.jsonl",
    )
    ev.add_argument(
        "--dry-run", action="store_true",
        help="Step through probes without making real API calls (pipeline test)",
    )
    ev.set_defaults(func=cmd_evaluate)

    # ── score ─────────────────────────────────────────────────────────────────
    sc = sub.add_parser("score", help="Score existing raw model outputs")
    sc.add_argument("--model", default=None, help="Score only this model's raw output")
    sc.add_argument("--cloudflare", action="store_true")
    sc.set_defaults(func=cmd_score)

    # ── profile ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("profile", help="Print G-MASS safety profile (CSR/SDS/RAR)")
    pr.add_argument("--model", default=None, help="Profile only this model")
    pr.add_argument("--save",  action="store_true", help="Save profiles to JSON")
    pr.set_defaults(func=cmd_profile)

    # ── report ────────────────────────────────────────────────────────────────
    rp = sub.add_parser("report", help="Generate Ghana Clinical Simulation Report")
    rp.set_defaults(func=cmd_report)

    # ── probe ─────────────────────────────────────────────────────────────────
    pb = sub.add_parser("probe", help="Probe set utilities")
    pb.add_argument(
        "probe_action",
        choices=["check", "stats"],
        help="check: validate required fields | stats: show domain/category breakdown",
    )
    pb.set_defaults(func=cmd_probe)

    # ── combine (internal / advanced) ────────────────────────────────────────
    cb = sub.add_parser("combine", help="Combine all scored outputs into one file")
    cb.set_defaults(func=cmd_combine)

    return parser


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Called by the `gmass` console script installed via pyproject.toml.
    Also the ENTRYPOINT for the Docker container.
    """
    parser = build_parser()
    args   = parser.parse_args()

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[G-MASS] Interrupted. Partial results are saved — re-run to resume.")
        sys.exit(0)
    except Exception as e:
        # Avoid crashing silently in nohup runs — always log before exit
        try:
            from medisafe_gh.core.logger import get_logger
            get_logger("gmass.cli").exception(f"Fatal error: {e}")
        except Exception:
            print(f"[G-MASS] Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
