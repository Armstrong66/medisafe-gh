# scorer/pipeline.py
# MediSafe-GH · G-MASS Project
# Team D — Engineering Lead
#
# Batch runner — reads GMASS_105-probes.jsonl, calls each model on every
# probe, classifies the response, and writes results to a CSV.
#
# Usage:
#   from scorer.pipeline import run_batch
#   run_batch(
#       probes_jsonl = "probes/GMASS_105-probes.jsonl",
#       output_csv   = "outputs/gpt4o_english.csv",
#       model        = "gpt4o",
#       delay_seconds = 1.5,
#   )

import csv
import json
import os
import time

from models.router import call_model
from scorer.classifiers import classify


def print_wrapped(text: str, prefix: str = "    ", width: int = 90):
    """Print text wrapped for readable terminal output."""
    words = str(text or "").split()
    if not words:
        print(f"{prefix}<empty>")
        return

    line = prefix
    for word in words:
        if len(line) + len(word) + 1 > width:
            print(line.rstrip())
            line = prefix + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line.rstrip())


def print_probe_result(probe: dict, response: str, labels: dict):
    """Print the full probe, model response, and safety classification."""
    print("    Probe:")
    print_wrapped(probe["prompt"])
    print("    Model response:")
    print_wrapped(response)
    print("    Safety scores / classification:")
    print(f"      Gemini classifier : {labels['label_gemini'].upper()}")
    print(f"      Rules classifier  : {labels['label_rules'].upper()}")
    print(f"      Classifiers agreed: {labels['agreed']}")
    print(f"      Final label       : {labels['final_label'].upper()}")
    print(f"      Needs review      : {labels['needs_review']}")
    print()


def print_saved_result(row: dict):
    """Print a previously saved CSV row in the same detailed format."""
    labels = {
        "label_gemini": row.get("label_gemini") or "n/a",
        "label_rules": row.get("label_rules") or "n/a",
        "agreed": row.get("agreed") or "n/a",
        "final_label": row.get("final_label") or "error",
        "needs_review": row.get("needs_review") or "n/a",
    }
    probe = {"prompt": row.get("prompt", "")}
    print(f"  [saved] {row.get('probe_id', '<unknown>')} -> {labels['final_label']}")
    print_probe_result(probe, row.get("response", ""), labels)


def load_probes(jsonl_path: str) -> list:
    """Load probes from a .jsonl file. Returns a list of probe dicts."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch(
    probes_jsonl: str,
    output_csv: str,
    model: str,
    delay_seconds: float = 1.5,
    resume: bool = True,
) -> list:
    """
    Run a single model over all probes in the JSONL file.

    Args:
        probes_jsonl  : path to GMASS_105-probes.jsonl
        output_csv    : where to write results
        model         : one of "gpt4o", "gemini", "llama", "phi3", "biomistral"
        delay_seconds : wait between API calls (avoids rate limits)
        resume        : if True, skip probes already in the output CSV

    Returns:
        List of result dicts (one per probe).
    """
    probes = load_probes(probes_jsonl)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)

    # ── Resume: find already-completed probe IDs ──────────────────────────────
    completed_ids = set()
    completed_rows = []
    if resume and os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            completed_rows = list(reader)
            completed_ids = {row["probe_id"] for row in completed_rows}
        if completed_ids:
            print(f"  Resuming — {len(completed_ids)} probes already done, skipping.")

    # ── CSV output columns ────────────────────────────────────────────────────
    FIELDNAMES = [
        "probe_id",
        "disease_domain",
        "failure_category",
        "language",
        "model",
        "prompt",
        "response",
        "label_gemini",
        "label_rules",
        "agreed",
        "final_label",
        "needs_review",
        "error",
    ]

    # Open CSV in append mode if resuming, write mode otherwise
    file_mode = "a" if completed_ids else "w"
    results = []

    with open(output_csv, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if file_mode == "w":
            writer.writeheader()

        total   = len(probes)
        pending = [p for p in probes if p["probe_id"] not in completed_ids]

        print(f"\nRunning: {model} | {len(pending)} probes to process")
        print(f"Output:  {output_csv}\n")

        if completed_rows:
            print("Previously saved results:")
            for row in completed_rows:
                print_saved_result(row)

        for i, probe in enumerate(pending, 1):
            probe_id         = probe["probe_id"]
            probe_text       = probe["prompt"]
            failure_category = probe["failure_category"]
            disease_domain   = probe["disease_domain"]
            language         = probe.get("language", "english")

            row = {
                "probe_id":         probe_id,
                "disease_domain":   disease_domain,
                "failure_category": failure_category,
                "language":         language,
                "model":            model,
                "prompt":           probe_text,
                "response":         "",
                "label_gemini":     "",
                "label_rules":      "",
                "agreed":           "",
                "final_label":      "error",
                "needs_review":     True,
                "error":            "",
            }

            try:
                # ── Call the model ─────────────────────────────────────────
                response = call_model(model, probe_text)
                row["response"] = response

                # ── Classify the response ─────────────────────────────────
                labels = classify(probe_text, response, failure_category)
                row.update({
                    "label_gemini": labels["label_gemini"],
                    "label_rules":  labels["label_rules"],
                    "agreed":       labels["agreed"],
                    "final_label":  labels["final_label"],
                    "needs_review": labels["needs_review"],
                    "error":        "",
                })

                status = "✓" if labels["final_label"] == "safe" else "✗ UNSAFE"
                print(f"  [{i:>3}/{len(pending)}] {probe_id:10s} → {labels['final_label']:6s}  {status}")
                print_probe_result(probe, response, labels)

            except Exception as e:
                row["error"] = str(e)[:300]
                print("    Probe:")
                print_wrapped(probe_text)
                print("    Model response:")
                print("    <none>")
                print("    Safety scores / classification:")
                print("      Final label       : ERROR")
                print()
                print(f"  [{i:>3}/{len(pending)}] {probe_id:10s} → ERROR: {str(e)[:80]}")

            writer.writerow(row)
            f.flush()  # write immediately — safe against crashes
            results.append(row)

            if i < len(pending):
                time.sleep(delay_seconds)

    passed  = sum(1 for r in results if r["final_label"] == "safe")
    failed  = sum(1 for r in results if r["final_label"] == "unsafe")
    errors  = sum(1 for r in results if r["error"])
    reviews = sum(1 for r in results if r["needs_review"] is True)

    print(f"\n  Done: {passed} safe | {failed} unsafe | {errors} errors | {reviews} need review")
    print(f"  Results saved → {output_csv}\n")

    return results
