"""report.py -- OWNER: Person 4. Orchestrator that assembles profiling_report.json."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import metadata as md, rules, stats, viz
from .io import load_dataset, maybe_sample
from .schemas import DatasetInfo, ProfilingReport

log = logging.getLogger("profiler")


def generate_report(df: pd.DataFrame, source_name: str = "dataset",
                    out_dir: str | Path = "output", embed_images: bool = False) -> ProfilingReport:
    """Run all engines on a DataFrame and return the validated report.
    Visualisation and rule failures become warnings instead of crashing the run."""
    n_rows, n_cols = df.shape
    df, sampled = maybe_sample(df)
    warnings: list[str] = []

    metadata = md.extract_metadata(df)
    statistics = stats.run_statistics(df, metadata)

    try:
        visuals = viz.generate_all(df, Path(out_dir), embed_images)
    except Exception as e:  # noqa: BLE001
        log.exception("viz failed")
        warnings.append(f"visualisations failed: {e}")
        visuals = []

    try:
        flags = rules.run_rules(df, metadata, statistics)
    except Exception as e:  # noqa: BLE001
        log.exception("rules failed")
        warnings.append(f"rule engine failed: {e}")
        flags = []

    return ProfilingReport(
        dataset=DatasetInfo(source_name=source_name, n_rows=n_rows, n_cols=n_cols,
                            sampled=sampled, sample_size=len(df) if sampled else None),
        metadata=metadata, statistics=statistics, visualisations=visuals,
        flags=flags, warnings=warnings)


def profile_file(path: str | Path, out_dir: str | Path = "output") -> Path:
    """Convenience: load a file, profile it, write profiling_report.json, return its path."""
    df = load_dataset(path)
    report = generate_report(df, source_name=Path(path).name, out_dir=out_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "profiling_report.json"
    target.write_text(report.model_dump_json(indent=2))
    return target
