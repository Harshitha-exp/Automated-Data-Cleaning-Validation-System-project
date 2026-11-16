"""Data-quality scoring (Person 2, Module 2: the data scientist's part).

Scores a dataset from 0 to 100 on four dimensions, so you can score the raw data, score the
cleaned data, and report the difference (the quality delta).

    completeness : share of cells that are not missing
    uniqueness   : share of rows that are not duplicates of another row
    validity     : share of values that fit the schema (type, min/max, allowed values, format)
    consistency  : share of text values with no stray spaces and no casing variants

Things to know when reading the numbers:
- Completeness goes up by construction after imputation. Look at validity and consistency to
  see whether the data really got better, and at how many cells were imputed.
- Validity is only scored for columns that have something that can fail: text declared as a
  number or date, a min/max, allowed values or a format. A number column with no min/max has
  nothing to check, so it is left out and never counted as perfect.
- Validity and consistency look at a random sample (quality.sample_size) to stay fast on big
  files. Completeness and uniqueness use every row.
- "Missing" uses the same null_markers as the rule engine (N/A, NA, null and so on).

Key functions:
    score_quality(df, schema=None, config=None) -> dict
    quality_delta(before, after) -> dict
    score_before_after(raw_df, cleaned_df, schema=None, config=None) -> dict
"""
import pandas as pd
from ds_config import get_config
from ds_schema_stub import basic_schema, col_info
from impute_stats import apply_null_markers

NUMERIC_TYPES = {"integer", "float"}
TRUE_FALSE = {"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}
DIMENSIONS = ("completeness", "uniqueness", "validity", "consistency")

def _is_texty(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)

def _validity(series, info):
    """Share of non-missing values that pass every rule the schema gives. None if no rules."""
    s = series.dropna()
    if s.empty:
        return None
    declared = info.get("type")
    rules = False                       # becomes True only when there is something that can fail
    ok = pd.Series(True, index=s.index)

    if declared in NUMERIC_TYPES:
        is_numeric = pd.api.types.is_numeric_dtype(s)
        num = s if is_numeric else pd.to_numeric(s, errors="coerce")
        if not is_numeric:              # text declared as a number: the type check can fail
            rules = True
            ok &= num.notna()
        if declared == "integer" and not pd.api.types.is_integer_dtype(s):
            rules = True
            ok &= (num == num.round()).fillna(False)
        if info.get("min") is not None:
            rules = True
            ok &= (num >= info["min"]).fillna(False)
        if info.get("max") is not None:
            rules = True
            ok &= (num <= info["max"]).fillna(False)
    elif declared == "datetime":
        if not pd.api.types.is_datetime64_any_dtype(s):
            rules = True
            try:
                parsed = pd.to_datetime(s, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                parsed = pd.to_datetime(s, errors="coerce")
            ok &= parsed.notna()
    elif declared == "boolean":
        if not pd.api.types.is_bool_dtype(s):
            rules = True
            ok &= s.astype(str).str.strip().str.lower().isin(TRUE_FALSE)

    if info.get("allowed_values"):
        rules = True
        ok &= s.astype(str).isin([str(v) for v in info["allowed_values"]])
    if info.get("format_regex"):
        rules = True
        ok &= s.astype(str).str.fullmatch(info["format_regex"]).fillna(False)

    return float(ok.mean()) if rules else None

def _consistency(series, max_ratio):
    """Share of text values with no stray spaces and no casing variants. None for non-text."""
    if not _is_texty(series):
        return None
    s = series.dropna().astype(str)
    if s.empty:
        return None
    bad = s != s.str.strip()
    if s.nunique() / len(s) <= max_ratio:          # only judge casing on repeated categories
        trimmed = s.str.strip()
        variants = trimmed.groupby(trimmed.str.lower()).transform("nunique") > 1
        bad = bad | variants
    return float(1 - bad.mean())

def _mean_of(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None

def score_quality(df, schema=None, config=None):
    """Score a dataset from 0 to 100. See the top of this file for what is measured."""
    cfg = get_config(config)
    qcfg = cfg["quality"]
    schema = schema or basic_schema(df)
    data = apply_null_markers(df, cfg["null_markers"])
    n_rows, n_cols = data.shape

    if n_rows == 0 or n_cols == 0:
        return {"overall": None, "dimensions": {d: None for d in DIMENSIONS},
                "weights_used": {}, "columns": {}, "n_rows": n_rows, "n_columns": n_cols,
                "n_missing_cells": 0, "n_duplicate_rows": 0, "sampled_rows": 0,
                "notes": ["empty dataset: nothing to score"]}

    missing = data.isna().sum()
    n_missing_cells = int(missing.sum())
    completeness = 1 - n_missing_cells / (n_rows * n_cols)

    subset = qcfg["duplicate_subset"]
    n_duplicates = int(data.duplicated(subset=subset).sum())
    uniqueness = 1 - n_duplicates / n_rows

    sample = data.sample(qcfg["sample_size"], random_state=0) if n_rows > qcfg["sample_size"] else data
    columns = {}
    for col in data.columns:
        columns[col] = {
            "completeness": float(1 - missing[col] / n_rows),
            "validity": _validity(sample[col], col_info(schema, col)),
            "consistency": _consistency(sample[col], qcfg["categorical_max_ratio"]),
        }
    scores = {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "validity": _mean_of([c["validity"] for c in columns.values()]),
        "consistency": _mean_of([c["consistency"] for c in columns.values()]),
    }

    weights = {d: qcfg["weights"].get(d, 0) for d in DIMENSIONS if scores[d] is not None}
    total = sum(weights.values())
    overall = (100 * sum(weights[d] * scores[d] for d in weights) / total) if total > 0 else None

    notes = []
    if scores["validity"] is None:
        notes.append("validity not scored: the schema has no rules (types, ranges, allowed values) to check")
    if scores["consistency"] is None:
        notes.append("consistency not scored: no text columns")
    return {
        "overall": None if overall is None else round(overall, 2),
        "dimensions": {d: None if v is None else round(v, 4) for d, v in scores.items()},
        "weights_used": {d: w / total for d, w in weights.items()} if total > 0 else {},
        "columns": {c: {k: None if v is None else round(v, 4) for k, v in vals.items()}
                    for c, vals in columns.items()},
        "n_rows": n_rows, "n_columns": n_cols,
        "n_missing_cells": n_missing_cells, "n_duplicate_rows": n_duplicates,
        "sampled_rows": len(sample), "notes": notes,
    }

def quality_delta(before, after):
    """Difference between two scores from score_quality (after minus before)."""
    dims, improved, worsened = {}, [], []
    for d in DIMENSIONS:
        b, a = before["dimensions"].get(d), after["dimensions"].get(d)
        if b is None or a is None:
            continue
        dims[d] = round(a - b, 4)
        if dims[d] > 0:
            improved.append(d)
        elif dims[d] < 0:
            worsened.append(d)
    b, a = before["overall"], after["overall"]
    return {
        "overall_before": b, "overall_after": a,
        "overall_delta": None if b is None or a is None else round(a - b, 2),
        "dimensions": dims, "improved": improved, "worsened": worsened,
    }

def score_before_after(raw_df, cleaned_df, schema=None, config=None):
    """Score the raw and cleaned data with the same schema and settings, and the difference."""
    schema = schema or basic_schema(raw_df)
    before = score_quality(raw_df, schema, config)
    after = score_quality(cleaned_df, schema, config)
    return {"before": before, "after": after, "delta": quality_delta(before, after)}