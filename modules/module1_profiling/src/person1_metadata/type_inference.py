"""
type_inference.py
-------------------
Person 1 — Metadata Extraction Engine: column TYPE inference.

Public function: infer_types(df) -> dict[str, dict]

This answers "what structural kind of data is in this column" — int,
float, bool, datetime, categorical, text — independent of what the
column *means* (that's semantics.py). Kept separate on purpose: Person 3's
rules, Person 2's distributions, and Person 4's schema all want the
structural type without caring whether a column is "an email" or "a
county name".
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# The six buckets named in the brief, plus "unknown" for all-null columns.
# Deliberately does NOT include "identifier" — "this column is an ID" is a
# question of MEANING, not structure, and belongs in semantics.py as
# semantic == "id" (that's also exactly the string Person 3's rule engine
# checks for: column_rules.get_semantic(metadata, col) == "id").
STRUCTURAL_TYPES = ("integer", "float", "boolean", "datetime", "categorical", "text", "unknown")

# A column is "categorical" rather than "text" by EITHER of two rules:
#   (a) few enough distinct values in absolute terms, full stop — a
#       10-row column with 5 repeated county names is obviously
#       categorical even though 5/10 = 50% looks like a "high" ratio;
#       ratio alone is misleading on small data.
#   (b) for larger columns where (a) doesn't apply, a small enough
#       cardinality RATIO with a higher absolute ceiling — e.g. a
#       million-row column with 150 distinct commodity names (ratio
#       0.00015) is still clearly categorical even though 150 > the
#       small-dataset absolute cap in rule (a).
# Two thresholds instead of one AND/OR mix, specifically to avoid the
# single-threshold trap: a flat "ratio <= 0.05" alone misclassifies small
# tables (5/10 fails it) and a flat "n_unique <= 50" alone misclassifies
# genuinely free-text columns in small tables (10 rows of 10 unique
# comments would wrongly pass n_unique<=50 too) — rule (b)'s ratio check
# is what keeps that case as "text".
CATEGORICAL_MAX_ABSOLUTE_SMALL = 50   # rule (a): candidate for categorical at or below this count
CATEGORICAL_SMALL_MAX_RATIO = 0.5     # rule (a) sanity floor: must have *some* repetition
CATEGORICAL_MAX_RATIO = 0.01          # rule (b): ratio ceiling once above the small-count cap
CATEGORICAL_MAX_ABSOLUTE_LARGE = 1000  # rule (b): absolute ceiling even if ratio is tiny


def _structural_type_of_series(series: pd.Series) -> str:
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"

    # object/string dtype: decide categorical vs free text by cardinality.
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"
    n = len(non_null)
    n_unique = non_null.nunique()
    ratio = n_unique / n if n else 0

    if n_unique <= CATEGORICAL_MAX_ABSOLUTE_SMALL and ratio <= CATEGORICAL_SMALL_MAX_RATIO:
        return "categorical"
    if ratio <= CATEGORICAL_MAX_RATIO and n_unique <= CATEGORICAL_MAX_ABSOLUTE_LARGE:
        return "categorical"
    return "text"


def infer_types(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Infer the structural type of every column in `df`.

    Returns
    -------
    {
        "<column_name>": {
            "dtype": "<pandas dtype as a string, e.g. 'int64'>",
            "type": "<one of STRUCTURAL_TYPES>",
            "n_unique": <int, non-null unique count>,
            "n_missing": <int>,
            "cardinality_ratio": <float, n_unique / n_non_null, 0.0 if empty>,
        },
        ...
    }
    """
    result: dict[str, dict[str, Any]] = {}
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        n_missing = int(n_rows - len(non_null))
        cardinality_ratio = (n_unique / len(non_null)) if len(non_null) else 0.0

        result[col] = {
            "dtype": str(series.dtype),
            "type": _structural_type_of_series(series),
            "n_unique": n_unique,
            "n_missing": n_missing,
            "cardinality_ratio": round(cardinality_ratio, 4),
        }
    return result
