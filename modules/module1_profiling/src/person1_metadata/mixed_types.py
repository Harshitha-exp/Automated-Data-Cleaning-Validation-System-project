"""
mixed_types.py
----------------
Person 1 — Metadata Extraction Engine: mixed-type column detection.

Public function: detect_mixed_types(df) -> dict[str, dict]

A "mixed-type" column is an object-dtype column whose values don't agree
on a single underlying type once you actually look at them — e.g. a
column that's 80% integers-as-strings and 20% free text, or a "notes"
field that occasionally holds a date. This is a different concept from
"categorical vs text" (type_inference.py) and from "what does this column
mean" (semantics.py): it's specifically a dirty-data signal that Module 2
(cleaning) needs before it tries to coerce the column to one dtype.

A column pandas already parsed as pure int64/float64/bool/datetime64
cannot be mixed-type by construction (pandas would have failed or
upcast to object otherwise), so only object-dtype columns are checked.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
    "%Y-%m-%d %H:%M:%S", "%d %b %Y", "%b %d, %Y",
)
_INTEGER_RE_CHARS = set("0123456789-")


def _looks_like_date(token: str) -> bool:
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(token, fmt)
            return True
        except ValueError:
            continue
    return False


def _classify_token(token: str) -> str:
    """Bucket a single string token into a coarse value-type class."""
    if token == "":
        return "empty"
    stripped = token.lstrip("-")
    if stripped.isdigit():
        return "integer"
    try:
        float(token)
        return "float"
    except ValueError:
        pass
    if _looks_like_date(token):
        return "date"
    if token.replace(".", "", 1).lstrip("-").isdigit():
        return "float"
    return "string"


def detect_mixed_types(df: pd.DataFrame, sample_size: int = 2000, mixed_threshold: float = 0.90) -> dict[str, dict[str, Any]]:
    """
    Detect, for every object-dtype column, whether it mixes incompatible
    value types.

    Parameters
    ----------
    df : pd.DataFrame
    sample_size : int
        Non-null values are sampled (not scanned in full) on large
        columns for speed; this is the cap.
    mixed_threshold : float
        A column is flagged mixed when no single value-type bucket covers
        at least this share of the sample AND more than one bucket has
        meaningful presence (>= 5% of the sample). This mirrors the
        wording in the brief ("mixed-type columns") rather than flagging
        every column that has a single stray value.

    Returns
    -------
    {
        "<column_name>": {
            "is_mixed_type": <bool>,
            "type_breakdown": {"integer": 0.8, "string": 0.2, ...},  # only for object dtype
        },
        ...
    }
    Non-object-dtype columns always get {"is_mixed_type": False, "type_breakdown": {}}.
    """
    result: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        series = df[col]
        # pandas >= 3.0 defaults string columns to a dedicated `str` dtype
        # instead of `object` — `dtype == object` alone misses them
        # entirely (silently returns "not mixed" for every text column).
        # is_object_dtype also still catches genuinely-mixed object columns
        # (e.g. int + str + NaN all in one column), which is exactly the
        # case this function exists to find.
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            result[col] = {"is_mixed_type": False, "type_breakdown": {}}
            continue

        non_null = series.dropna()
        if non_null.empty:
            result[col] = {"is_mixed_type": False, "type_breakdown": {}}
            continue

        sample = (
            non_null.sample(sample_size, random_state=42)
            if len(non_null) > sample_size
            else non_null
        )

        buckets: dict[str, int] = {}
        for v in sample:
            bucket = _classify_token(str(v).strip())
            buckets[bucket] = buckets.get(bucket, 0) + 1

        total = sum(buckets.values())
        breakdown = {k: round(v / total, 4) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}

        dominant_share = max(breakdown.values()) if breakdown else 1.0
        n_significant_buckets = sum(1 for p in breakdown.values() if p >= 0.05)
        is_mixed = dominant_share < mixed_threshold and n_significant_buckets > 1

        result[col] = {"is_mixed_type": is_mixed, "type_breakdown": breakdown}

    return result
