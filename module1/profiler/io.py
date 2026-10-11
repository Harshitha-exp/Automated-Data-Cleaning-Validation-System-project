"""io.py -- OWNER: Person 4. Dataset loading and sampling."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MAX_ROWS_FOR_PROFILING = 1_000_000


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load CSV / TSV / Excel / Parquet / JSON into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            return pd.read_csv(path, sep=sep, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=sep, low_memory=False, encoding="latin-1")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def maybe_sample(df: pd.DataFrame, max_rows: int = MAX_ROWS_FOR_PROFILING, seed: int = 42):
    """Return (df_or_sample, was_sampled)."""
    if len(df) <= max_rows:
        return df, False
    return df.sample(max_rows, random_state=seed), True
