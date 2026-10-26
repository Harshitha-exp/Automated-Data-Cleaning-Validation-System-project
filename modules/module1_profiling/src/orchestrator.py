"""
orchestrator.py
------------------
Person 4 — Integration & Orchestration.

Runs the full Module 1 pipeline end to end: Person 1's metadata engine,
then Person 2's statistics + visualisations, then Person 3's rule engine
(fed Person 1's metadata, as their interface requires), and combines all
three into the single profiling_report.json structure this module ships.

Key function: generate_report(df, ...) -> dict

Each stage is wrapped so one module's failure doesn't take down the whole
report: if, say, Person 2's correlation pass throws on a pathological
column, the report still comes back with Person 1 and Person 3's results
intact plus a description of what failed in "errors" — the same
fail-soft shape Person 3's own run_rules() already uses internally.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0.0"

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PERSON3_DIR = os.path.join(_SRC_DIR, "person3_rules")


def _load_person3_engine():
    """
    Person 3's engine.py (and the column_rules/pii/formats modules it
    imports) uses flat, non-relative imports — written and tested as a
    standalone script directory, not a package. Rather than editing their
    source to fit a package layout (and risk drifting from the code they
    actually tested), this puts their directory directly on sys.path, the
    same way their own test suite's conftest.py does, and imports it as a
    plain top-level module.
    """
    if _PERSON3_DIR not in sys.path:
        sys.path.insert(0, _PERSON3_DIR)
    import engine as person3_engine  # noqa: PLC0415
    return person3_engine


def generate_report(
    df: pd.DataFrame,
    *,
    include_visualisations: bool = True,
    rules_config: dict | None = None,
) -> dict[str, Any]:
    """
    Run Person 1 -> Person 2 -> Person 3 over `df` and return the combined
    profiling_report.json structure.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.
    include_visualisations : bool
        Set False to skip chart rendering (metadata + statistics + flags
        only) — useful for a quick pass over a very large file, since the
        charts are the most expensive step.
    rules_config : dict | None
        Passed straight through to Person 3's run_rules(config=...); None
        uses their shipped rules.yaml defaults.

    Returns
    -------
    {
        "schema_version": "1.0.0",
        "generated_at": "<ISO 8601 timestamp>",
        "n_rows": int, "n_columns": int,
        "metadata": {...},        # Person 1's build_metadata() output
        "statistics": {...},      # Person 2's compute_statistics() output
        "visualisations": {...},  # Person 2's generate_visualisations() output, or {} if skipped
        "flags": [...], "total_flags": int,   # Person 3's run_rules() output
        "errors": [ {"stage": "...", "error": "..."}, ... ],
    }
    """
    from person1_metadata.metadata import build_metadata
    from person2_profiling.charts import generate_visualisations
    from person2_profiling.stats import compute_statistics

    errors: list[dict[str, str]] = []

    metadata: dict[str, Any] = {}
    try:
        metadata = build_metadata(df)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: one stage must not sink the report
        errors.append({"stage": "person1_metadata", "error": f"{type(exc).__name__}: {exc}"})

    statistics: dict[str, Any] = {}
    try:
        statistics = compute_statistics(df, metadata=metadata or None)
    except Exception as exc:  # noqa: BLE001
        errors.append({"stage": "person2_statistics", "error": f"{type(exc).__name__}: {exc}"})

    visualisations: dict[str, Any] = {}
    if include_visualisations:
        try:
            correlations = statistics.get("correlations")
            visualisations = generate_visualisations(df, correlations=correlations)
        except Exception as exc:  # noqa: BLE001
            errors.append({"stage": "person2_visualisations", "error": f"{type(exc).__name__}: {exc}"})

    flags: list[dict[str, Any]] = []
    total_flags = 0
    try:
        person3_engine = _load_person3_engine()
        rule_result = person3_engine.run_rules(df, metadata=metadata or None, config=rules_config)
        flags = rule_result.get("flags", [])
        total_flags = rule_result.get("total_flags", len(flags))
        # Person 3's own per-rule errors (a single rule crashing) are
        # surfaced too, tagged so it's clear they came from inside their
        # engine rather than from the orchestrator itself.
        for rule_error in rule_result.get("errors", []):
            errors.append({"stage": "person3_rules", "error": str(rule_error)})
    except Exception as exc:  # noqa: BLE001
        errors.append({
            "stage": "person3_rules",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=3),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "metadata": metadata,
        "statistics": statistics,
        "visualisations": visualisations,
        "flags": flags,
        "total_flags": total_flags,
        "errors": errors,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Module 1 — run the full profiling pipeline on a dataset.")
    parser.add_argument("dataset", help="Path to a .csv/.tsv/.xlsx/.parquet file")
    parser.add_argument("--out", default="profiling_report.json", help="Where to write the report JSON")
    parser.add_argument("--no-visuals", action="store_true", help="Skip chart rendering for a faster pass")
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

    report = generate_report(frame, include_visualisations=not args.no_visuals)
    print(f"Profiled {report['n_rows']} rows x {report['n_columns']} columns")
    print(f"Flags: {report['total_flags']}  Errors: {len(report['errors'])}")
    for err in report["errors"]:
        print(f"  [{err['stage']}] {err['error']}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWritten to {args.out}")
