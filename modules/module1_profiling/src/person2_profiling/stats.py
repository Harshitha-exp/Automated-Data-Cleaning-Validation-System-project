"""
person2_profiling/stats.py
----------------------------
Person 2 — Profiling Engine (the statistics half of "Profiling Engine and
Visualisations"). This is a from-scratch implementation written for the
"final GitHub module" integration — no reusable Person 2 functions had
been submitted at the time this was built (only an auto-generated
ydata-profiling HTML/JSON report, which bundles everything internally and
doesn't expose the individual functions the brief names). If Person 2
delivers their own version later, swap this file out — every function
below keeps the exact signature the brief lists so the orchestrator
doesn't need to change.

Key functions (as named in the task breakdown):
    profile_missing(df)          -> missing-value matrix, summarised
    type_consistency_check(df)   -> does each column's data match its declared type
    compute_cardinality(df)      -> unique-value / cardinality analysis
    compute_distributions(df)    -> per-column distribution stats
    compute_correlations(df)     -> numeric (Pearson) + categorical (Cramer's V)

compute_statistics(df, metadata=None) combines all five into the single
"statistics" section of profiling_report.json, the same way Person 1's
build_metadata() and Person 3's run_rules() are each one combining call.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Columns above this cardinality are skipped by the categorical-correlation
# pass (Cramer's V is O(n_categories^2) memory via the contingency table,
# and a free-text column with thousands of distinct values is never
# meaningfully "correlated" with anything in a categorical sense anyway).
MAX_CATEGORICAL_LEVELS_FOR_CORR = 50
# Row sampling cap for correlation computation on huge datasets — a
# million-row Pearson/Cramer's V pass is unnecessary; a large random
# sample gives a stable estimate at a fraction of the cost.
CORRELATION_SAMPLE_ROWS = 100_000


def profile_missing(df: pd.DataFrame) -> dict[str, Any]:
    """
    Summarise the missing-value matrix as statistics (the matrix itself,
    as a picture, is charts.plot_missing_heatmap's job — returning a
    literal row x column boolean grid here would be enormous and useless
    as JSON on a million-row file).

    Returns
    -------
    {
        "per_column": {"<col>": {"n_missing": int, "pct_missing": float}, ...},
        "total_cells": int,
        "total_missing_cells": int,
        "pct_missing_overall": float,
        "rows_with_any_missing": int,
        "pct_rows_with_any_missing": float,
    }
    """
    n_rows, n_cols = df.shape
    null_mask = df.isnull()

    per_column = {}
    for col in df.columns:
        n_missing = int(null_mask[col].sum())
        per_column[col] = {
            "n_missing": n_missing,
            "pct_missing": round(100 * n_missing / n_rows, 4) if n_rows else 0.0,
        }

    total_cells = n_rows * n_cols
    total_missing = int(null_mask.to_numpy().sum())
    rows_with_any_missing = int(null_mask.any(axis=1).sum())

    return {
        "per_column": per_column,
        "total_cells": total_cells,
        "total_missing_cells": total_missing,
        "pct_missing_overall": round(100 * total_missing / total_cells, 4) if total_cells else 0.0,
        "rows_with_any_missing": rows_with_any_missing,
        "pct_rows_with_any_missing": round(100 * rows_with_any_missing / n_rows, 4) if n_rows else 0.0,
    }


def _declared_type_for(col: str, series: pd.Series, metadata: dict | None) -> str:
    """Prefer Person 1's declared structural type when metadata is
    available (this is the whole point of a shared pipeline); fall back
    to a plain pandas dtype guess otherwise so this function still works
    standalone."""
    if metadata:
        col_meta = metadata.get("columns", {}).get(col)
        if col_meta and col_meta.get("type"):
            return col_meta["type"]
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "text"


def _matches_declared_type(value: Any, declared_type: str) -> bool:
    if declared_type in ("integer", "float"):
        try:
            float(value)
            return True
        except (TypeError, ValueError):
            return False
    if declared_type == "boolean":
        return str(value).strip().lower() in {"true", "false", "0", "1", "yes", "no"}
    if declared_type == "datetime":
        return bool(pd.notna(pd.to_datetime(str(value), errors="coerce")))
    # categorical / text / unknown: any non-null string value counts as
    # consistent — there is no narrower "correct shape" to check here.
    return True


def type_consistency_check(
    df: pd.DataFrame, metadata: dict | None = None, sample_size: int = 2000
) -> dict[str, Any]:
    """
    For every column, check whether its *actual* values agree with its
    *declared* structural type (from Person 1's metadata when supplied).
    A pandas-typed numeric/datetime/boolean column is consistent by
    construction (pandas would not have parsed it otherwise); this is
    mainly informative for object/string-dtype columns that claim to be
    one thing but contain a mix.

    Returns
    -------
    {"<col>": {
        "declared_type": str,
        "consistency_ratio": float,   # share of sampled non-null values matching
        "is_consistent": bool,        # ratio >= 0.95
        "sample_size": int,
    }, ...}
    """
    result: dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        declared_type = _declared_type_for(col, series, metadata)
        non_null = series.dropna()

        if non_null.empty:
            result[col] = {
                "declared_type": declared_type, "consistency_ratio": 1.0,
                "is_consistent": True, "sample_size": 0,
            }
            continue

        # Already-parsed numeric/bool/datetime dtypes can't disagree with
        # themselves — pandas enforces that at parse time. Only
        # object/string columns need an actual per-value check.
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            result[col] = {
                "declared_type": declared_type, "consistency_ratio": 1.0,
                "is_consistent": True, "sample_size": int(len(non_null)),
            }
            continue

        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        matches = sum(1 for v in sample if _matches_declared_type(v, declared_type))
        ratio = matches / len(sample)
        result[col] = {
            "declared_type": declared_type,
            "consistency_ratio": round(ratio, 4),
            "is_consistent": ratio >= 0.95,
            "sample_size": int(len(sample)),
        }
    return result


def compute_cardinality(df: pd.DataFrame, top_n: int = 5) -> dict[str, Any]:
    """
    Unique-value / cardinality analysis per column.

    Returns
    -------
    {"<col>": {
        "n_unique": int, "cardinality_ratio": float,
        "is_constant": bool, "is_unique": bool,
        "top_values": [{"value": ..., "count": int}, ...],   # up to top_n
    }, ...}
    """
    n_rows = len(df)
    result: dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        ratio = (n_unique / len(non_null)) if len(non_null) else 0.0

        value_counts = non_null.value_counts().head(top_n)
        top_values = [
            {"value": str(idx), "count": int(cnt)} for idx, cnt in value_counts.items()
        ]

        result[col] = {
            "n_unique": n_unique,
            "cardinality_ratio": round(ratio, 4),
            "is_constant": n_unique <= 1,
            "is_unique": n_rows > 0 and n_unique == len(non_null) == n_rows,
            "top_values": top_values,
        }
    return result


def compute_distributions(df: pd.DataFrame) -> dict[str, Any]:
    """
    Per-column distribution summary. Numeric columns get the standard
    descriptive-stats set plus skew/kurtosis; everything else gets a
    frequency-shaped summary instead (mode + its share), since "mean" is
    meaningless for a categorical/text column.

    Returns
    -------
    {"<col>": {"kind": "numeric", "mean":.., "std":.., "min":.., "p25":..,
                "median":.., "p75":.., "max":.., "skew":.., "kurtosis":..}, ...}
    or
    {"<col>": {"kind": "categorical", "mode": .., "mode_share": float,
                "n_unique": int}, ...}
    """
    result: dict[str, Any] = {}
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if non_null.empty:
                result[col] = {"kind": "numeric"}
                continue
            desc = non_null.describe()
            result[col] = {
                "kind": "numeric",
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(desc["std"]), 4) if pd.notna(desc["std"]) else 0.0,
                "min": float(desc["min"]),
                "p25": float(desc["25%"]),
                "median": float(desc["50%"]),
                "p75": float(desc["75%"]),
                "max": float(desc["max"]),
                "skew": round(float(non_null.skew()), 4) if len(non_null) > 2 else 0.0,
                "kurtosis": round(float(non_null.kurtosis()), 4) if len(non_null) > 3 else 0.0,
            }
        else:
            if non_null.empty:
                result[col] = {"kind": "categorical"}
                continue
            counts = non_null.astype(str).value_counts()
            mode_value = counts.index[0]
            result[col] = {
                "kind": "categorical",
                "mode": mode_value,
                "mode_share": round(float(counts.iloc[0] / len(non_null)), 4),
                "n_unique": int(non_null.nunique()),
            }
    return result


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def _categorical_columns_for_corr(df: pd.DataFrame, max_levels: int) -> list[str]:
    cols = []
    for c in df.columns:
        series = df[c]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        if non_null.nunique() <= max_levels:
            cols.append(c)
    return cols


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series (Bergsma
    2013 correction) — the standard way to express categorical-categorical
    'correlation' as a single 0-1 number, the categorical analogue of
    Pearson's r the brief asks for."""
    contingency = pd.crosstab(a, b)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def compute_correlations(df: pd.DataFrame, sample_rows: int = CORRELATION_SAMPLE_ROWS) -> dict[str, Any]:
    """
    Correlation matrices for both numeric and categorical columns.

    Numeric <-> numeric uses Pearson's r (pandas' own .corr()).
    Categorical <-> categorical uses bias-corrected Cramer's V, restricted
    to columns with a sane number of distinct levels (see
    MAX_CATEGORICAL_LEVELS_FOR_CORR) — a free-text column has no
    meaningful "correlation" in this sense and would be O(n^2) to even
    build the contingency table for.

    On large datasets both passes run on a random row sample
    (`sample_rows`) rather than the full file — correlation strength is a
    population-level estimate, not something that needs every row, and
    this keeps the function usable on multi-million-row inputs.

    Returns
    -------
    {
        "numeric": {"columns": [...], "matrix": [[...], ...]},
        "categorical": {"columns": [...], "matrix": [[...], ...]},
    }
    Either key maps to an empty columns/matrix if fewer than 2 eligible
    columns exist for that kind.
    """
    frame = df.sample(sample_rows, random_state=42) if len(df) > sample_rows else df

    numeric_cols = _numeric_columns(frame)
    if len(numeric_cols) >= 2:
        pearson = frame[numeric_cols].corr(method="pearson")
        numeric_section = {
            "columns": list(pearson.columns),
            "matrix": [[round(float(v), 4) if pd.notna(v) else None for v in row] for row in pearson.to_numpy()],
        }
    else:
        numeric_section = {"columns": numeric_cols, "matrix": []}

    cat_cols = _categorical_columns_for_corr(frame, MAX_CATEGORICAL_LEVELS_FOR_CORR)
    if len(cat_cols) >= 2:
        matrix = [[1.0 if i == j else 0.0 for j in range(len(cat_cols))] for i in range(len(cat_cols))]
        for i, j in combinations(range(len(cat_cols)), 2):
            v = _cramers_v(frame[cat_cols[i]], frame[cat_cols[j]])
            matrix[i][j] = matrix[j][i] = round(v, 4)
        categorical_section = {"columns": cat_cols, "matrix": matrix}
    else:
        categorical_section = {"columns": cat_cols, "matrix": []}

    return {"numeric": numeric_section, "categorical": categorical_section}


def compute_statistics(df: pd.DataFrame, metadata: dict | None = None) -> dict[str, Any]:
    """
    Combine every statistic above into the single "statistics" section of
    profiling_report.json — the one function the orchestrator calls,
    mirroring Person 1's build_metadata() and Person 3's run_rules().
    """
    return {
        "missing": profile_missing(df),
        "type_consistency": type_consistency_check(df, metadata=metadata),
        "cardinality": compute_cardinality(df),
        "distributions": compute_distributions(df),
        "correlations": compute_correlations(df),
    }
