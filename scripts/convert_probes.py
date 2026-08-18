"""
scripts/convert_probes.py

Convert probe CSV/XLSX to canonical JSONL for the G-MASS pipeline.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except Exception:
    pd = None

CANONICAL_FIELDS = [
    "probe_id",
    "disease_domain",
    "failure_category",
    "english_prompt",
    "twi_prompt",
    "prompt_twi_validated",
    "ghanaian_en_prompt",
    "translation_status",
]

COMMON_COLUMN_MAP = {
    "id": "probe_id",
    "probe": "probe_id",
    "prompt": "english_prompt",
    "english_prompt": "english_prompt",
    "english prompt": "english_prompt",
    "twi_prompt": "twi_prompt",
    "twi prompt": "twi_prompt",
    "prompt_twi_validated": "prompt_twi_validated",
    "final_approved_twi": "twi_prompt",
    "ghanaian_en_prompt": "ghanaian_en_prompt",
    "final_approved_ghanaian_english": "ghanaian_en_prompt",
    "failure_category": "failure_category",
    "disease_domain": "disease_domain",
}


def normalize_text(s: object) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "")
    s = s.strip()
    s = unicodedata.normalize("NFKC", s)
    return s


def map_columns(columns: List[str]) -> Dict[str, str]:
    col_map: Dict[str, str] = {}
    for col in columns:
        low = str(col).lower()
        if low in COMMON_COLUMN_MAP:
            col_map[col] = COMMON_COLUMN_MAP[low]
            continue
        if "probe" in low and "id" in low:
            col_map[col] = "probe_id"
        elif "english" in low and "prompt" in low:
            col_map[col] = "english_prompt"
        elif "twi" in low and "prompt" in low:
            col_map[col] = "twi_prompt"
        elif "ghana" in low or "ghanaian" in low:
            col_map[col] = "ghanaian_en_prompt"
        elif "domain" in low:
            col_map[col] = "disease_domain"
        elif "failure" in low or "category" in low:
            col_map[col] = "failure_category"
    return col_map


def canonicalize_row(row: Dict[str, object], col_map: Dict[str, str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for src_col, value in row.items():
        if src_col in col_map:
            out[col_map[src_col]] = normalize_text(value)
    if not out.get("twi_prompt") and out.get("prompt_twi_validated"):
        out["twi_prompt"] = out.get("prompt_twi_validated")
    return out


def dataframe_to_jsonl(df, out_path: Path, strict: bool = False) -> int:
    required = ["probe_id", "english_prompt"]
    col_map = map_columns(list(df.columns))
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, row in df.iterrows():
            row_dict = {col: row[col] for col in df.columns}
            rec = canonicalize_row(row_dict, col_map)
            missing = [r for r in required if not rec.get(r)]
            if missing:
                msg = f"Row {i+1} missing required fields: {missing}"
                if strict:
                    raise ValueError(msg)
                else:
                    print("WARNING:", msg)
                    continue
            out_rec = {k: rec.get(k) for k in CANONICAL_FIELDS if rec.get(k) is not None}
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1
    return written


def load_input(path: Path, sheet: str | None = None):
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        if pd is None:
            raise EnvironmentError("pandas is required to read CSV/TSV files")
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if suffix in (".xlsx", ".xls"):
        if pd is None:
            raise EnvironmentError("pandas is required to read Excel files")
        return pd.read_excel(path, sheet_name=sheet or 0, dtype=str)
    if suffix in (".jsonl", ".ndjson"):
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
        import pandas as _pd
        return _pd.DataFrame(records)
    raise ValueError(f"Unsupported input file type: {path.suffix}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert probe CSV/XLSX to canonical JSONL")
    p.add_argument("input", help="Input file (csv, xlsx, jsonl)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--sheet", default=None, help="Excel sheet name or index")
    p.add_argument("--strict", action="store_true", help="Fail on first missing required field")
    args = p.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.out)

    if not in_path.exists():
        print(f"Input file not found: {in_path}")
        return 2

    if pd is None:
        print("ERROR: pandas is required for convert_probes.py. Install with: pip install pandas openpyxl")
        return 3

    df = load_input(in_path, args.sheet)
    df.columns = [str(c).strip() for c in df.columns]

    written = dataframe_to_jsonl(df, out_path, strict=args.strict)
    print(f"Wrote {written} records to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
