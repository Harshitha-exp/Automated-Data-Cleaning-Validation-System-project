"""
metadata.py
-------------
Person 1 — Metadata Extraction Engine: orchestrator.

Public function: build_metadata(df) -> dict

Combines type_inference.infer_types, semantics.detect_semantics, and
mixed_types.detect_mixed_types into the single structure the rest of the
team consumes directly:

    metadata = build_metadata(df)
    metadata["columns"]["Subsidy (US$)"]["semantic"]   # -> "currency"

This exact shape — metadata["columns"][col]["semantic"] — is the contract
Person 3's engine.run_rules(df, metadata) reads from (confirmed against
their actual column_rules.get_semantic()), and it's also what Person 4
should embed as the "metadata" section of profiling_report.json.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .mixed_types import detect_mixed_types
from .semantics import detect_semantics
from .type_inference import infer_types

SCHEMA_VERSION = "1.0.0"


def build_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the full metadata-extraction engine over every column of `df`.

    Returns
    -------
    {
        "schema_version": "1.0.0",
        "n_rows": <int>,
        "n_columns": <int>,
        "columns": {
            "<column_name>": {
                "dtype": "int64",
                "type": "integer",              # see type_inference.STRUCTURAL_TYPES
                "semantic": "id",                # see semantics.py docstring, or None
                "semantic_confidence": 0.85,
                "semantic_basis": ["column_name"],
                "is_mixed_type": False,
                "type_breakdown": {},
                "n_unique": 22,
                "n_missing": 0,
                "cardinality_ratio": 0.0,
            },
            ...
        }
    }
    """
    types = infer_types(df)
    semantics = detect_semantics(df, types=types)
    mixed = detect_mixed_types(df)

    columns: dict[str, Any] = {}
    for col in df.columns:
        t = types[col]
        s = semantics[col]
        m = mixed[col]
        columns[col] = {
            "dtype": t["dtype"],
            "type": t["type"],
            "semantic": s["semantic"],
            "semantic_confidence": round(s["confidence"], 3),
            "semantic_basis": s["basis"],
            "is_mixed_type": m["is_mixed_type"],
            "type_breakdown": m["type_breakdown"],
            "n_unique": t["n_unique"],
            "n_missing": t["n_missing"],
            "cardinality_ratio": t["cardinality_ratio"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": columns,
    }


# Re-exported so `from metadata import infer_types, detect_semantics,
# detect_mixed_types, build_metadata` gives every key function from one
# place, matching the brief's "Key functions" list verbatim.
__all__ = ["infer_types", "detect_semantics", "detect_mixed_types", "build_metadata"]


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Person 1 — run metadata extraction on a dataset.")
    parser.add_argument("dataset", help="Path to a .csv/.tsv/.xlsx/.parquet file")
    parser.add_argument("--out", default=None, help="Write the metadata dict to this JSON file")
    args = parser.parse_args()

    ext = os.path.splitext(args.dataset)[1].lower()
    if ext == ".csv":
        frame = pd.read_csv(args.dataset, low_memory=False)
    elif ext in (".xlsx", ".xlsm"):
        frame = pd.read_excel(args.dataset, engine="openpyxl")
    elif ext == ".parquet":
        frame = pd.read_parquet(args.dataset)
    else:
        raise SystemExit(f"Unsupported extension: {ext}")

    report = build_metadata(frame)
    print(f"Profiled {report['n_rows']} rows x {report['n_columns']} columns")
    for name, info in report["columns"].items():
        print(f"  {name:28s} type={info['type']:12s} semantic={str(info['semantic']):16s} "
              f"conf={info['semantic_confidence']:.2f} mixed={info['is_mixed_type']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWritten to {args.out}")
