"""
metadata.py -- OWNER: Person 1.

Contract: every function takes a DataFrame/Series and returns schema models.
The stubs below WORK (basic version) so the pipeline runs; replace with real logic.
"""
from __future__ import annotations

import re

import pandas as pd

from .schemas import ColumnMetadata, InferredType, Metadata, SemanticType

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def infer_column_type(series: pd.Series) -> tuple[InferredType, dict[str, float]]:
    """Infer a column's logical type and the share of values matching each type.

    TODO(P1): booleans-as-strings ("yes"/"no"), dates in many formats,
    numbers with currency symbols/thousand separators, categorical vs text threshold.
    """
    s = series.dropna()
    if s.empty:
        return "empty", {}
    if pd.api.types.is_bool_dtype(series):
        return "boolean", {"boolean": 1.0}
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", {"datetime": 1.0}
    num = pd.to_numeric(s.astype(str), errors="coerce")
    num_share = float(num.notna().mean())
    dist = {"numeric": round(num_share, 4), "text": round(1 - num_share, 4)}
    if num_share == 1.0:
        return ("integer" if (num % 1 == 0).all() else "float"), dist
    return ("categorical" if s.nunique() / len(s) < 0.5 else "text"), dist


def detect_semantic_type(series: pd.Series, name: str) -> tuple[SemanticType, float]:
    """Return (semantic_type, confidence). Combine name hints + value patterns.

    TODO(P1): phone, url, zipcode, currency, id, name, date, ip, etc.
    """
    s = series.dropna().astype(str).head(1000)
    if not s.empty and s.map(lambda v: bool(EMAIL_RE.match(v))).mean() > 0.8:
        return "email", 0.95
    if "email" in name.lower():
        return "email", 0.6
    return "none", 0.0


def detect_mixed_types(series: pd.Series) -> tuple[bool, dict[str, float]]:
    """True if a meaningful share (>5%) of values belong to a different type than the majority.

    TODO(P1): per-value type tagging (int / float / date / bool / str).
    """
    s = series.dropna()
    if s.empty:
        return False, {}
    num_share = float(pd.to_numeric(s.astype(str), errors="coerce").notna().mean())
    return (0.05 < num_share < 0.95), {"numeric": round(num_share, 4), "text": round(1 - num_share, 4)}


def extract_metadata(df: pd.DataFrame) -> Metadata:
    """Entry point called by report.generate_report()."""
    cols = []
    for i, name in enumerate(df.columns):
        itype, dist = infer_column_type(df[name])
        stype, conf = detect_semantic_type(df[name], str(name))
        mixed, _ = detect_mixed_types(df[name])
        cols.append(ColumnMetadata(
            name=str(name), position=i, pandas_dtype=str(df[name].dtype),
            inferred_type=itype, semantic_type=stype, semantic_confidence=conf,
            is_mixed_type=mixed, type_distribution=dist,
        ))
    return Metadata(n_rows=len(df), n_cols=df.shape[1],
                    memory_bytes=int(df.memory_usage(deep=True).sum()), columns=cols)
