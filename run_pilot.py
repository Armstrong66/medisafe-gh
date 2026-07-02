# run_pilot.py
# MediSafe-GH · G-MASS Project
#
# Runs a 15-probe pilot (5 per domain) through one model to validate
# the full pipeline before the 4,500-call Phase 3 run.
#
# Usage:
#   python run_pilot.py              # runs pilot on gemini (default)
#   python run_pilot.py gpt4o        # runs pilot on specific model

import json
import os
import sys

from scorer.pipeline import run_batch
from scorer.metrics  import compute_metrics, print_metrics, save_metrics_json
from models.router import normalize_model_name

PROBE_FILE = "probes/GMASS_105-probes.jsonl"
PILOT_FILE = "probes/pilot_15.jsonl"
MODEL      = normalize_model_name(sys.argv[1]) if len(sys.argv) > 1 else "gemini"

# ── Build 15-probe pilot (5 per domain) ──────────────────────────────────────
if not os.path.exists(PILOT_FILE):
    with open(PROBE_FILE, "r", encoding="utf-8") as f:
        all_probes = [json.loads(line) for line in f if line.strip()]

    # Pick 5 probes per domain (first 5 of each)
    pilot_probes = []
    domain_counts = {}
    for p in all_probes:
        d = p["disease_domain"]
        if domain_counts.get(d, 0) < 5:
            pilot_probes.append(p)
            domain_counts[d] = domain_counts.get(d, 0) + 1

    with open(PILOT_FILE, "w", encoding="utf-8") as f:
        for p in pilot_probes:
            f.write(json.dumps(p) + "\n")

    print(f"Created pilot file: {PILOT_FILE} ({len(pilot_probes)} probes)")

output_csv = f"outputs/pilot_{MODEL}.csv"
os.makedirs("outputs", exist_ok=True)

print(f"\nRunning 15-probe pilot on: {MODEL.upper()}")
print("This validates the full pipeline before the 4,500-call Phase 3 run.\n")

results = run_batch(
    probes_jsonl  = PILOT_FILE,
    output_csv    = output_csv,
    model         = MODEL,
    delay_seconds = 2.0,
)

# ── Compute and display metrics ───────────────────────────────────────────────
metrics = compute_metrics(output_csv)
print_metrics(metrics)
save_metrics_json(metrics, f"outputs/pilot_{MODEL}_metrics.json")

# ── Check for errors ──────────────────────────────────────────────────────────
errors = [r for r in results if r.get("error")]
if errors:
    print(f"\n  ⚠  {len(errors)} errors encountered:")
    for r in errors:
        print(f"    {r['probe_id']}: {r['error'][:100]}")
    print("\n  Fix errors before running the full Phase 3 evaluation.")
else:
    print(f"\n  ✓ Pilot complete — zero errors.")
    print(f"  Ready for full evaluation: python run_evaluation.py")
