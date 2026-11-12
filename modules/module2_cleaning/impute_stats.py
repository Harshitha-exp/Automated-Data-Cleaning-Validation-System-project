"""Statistical imputation (Person 2, Module 2: the data scientist's part).

Fills missing values with simple statistics: median or mean for numbers, most frequent
value for categories, optionally from similar rows (group-based). This is the BASELINE
that the ML imputers (KNN, regression, iterative) have to beat.

Rules this file follows:
- It never changes the dataframe you give it. It returns a cleaned copy plus a log.
- It does not invent identifiers, personal data, dates or free text: those columns are
  skipped and the skip is logged.
- Every change goes into the log as a plain dict (see _log_entry), so the orchestrator
  can add it to cleaning_log.json. Fill values are only logged for non-personal columns.
- If one column fails, the others are still processed and the failure is logged.

Key functions:
    impute_statistical(df, schema=None, config=None, columns=None) -> (cleaned_df, log)
    evaluate_statistical(df, schema=None, config=None, columns=None, ...) -> {column: scores}
    mask_known_values(df, column, share, seed) -> (masked_df, truth, masked_index)
"""
import numpy as np
import pandas as pd
from ds_config import get_config
from ds_schema_stub import basic_schema, col_info

STEP = "impute_statistical"
NUMERIC_TYPES = {"integer", "float"}

def _log_entry(column, action, **detail):
    """One line of the cleaning log. Proposed format, to be agreed with the log owner."""
    return {"step": STEP, "column": column, "action": action, **detail}

def apply_null_markers(df, markers):
    """Copy of df where marker values (like 'N/A' or -999) are real missing values."""
    out = df.copy()
    if not markers:
        return out
    str_markers = [str(m) for m in markers]
    num_markers = []
    for m in markers:
        try:
            num_markers.append(float(m))
        except (TypeError, ValueError):
            pass
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            if num_markers:
                out[col] = s.where(~s.isin(num_markers))
        elif not pd.api.types.is_datetime64_any_dtype(s):
            mask = s.astype("string").str.strip().isin(str_markers).fillna(False).to_numpy(dtype=bool)
            if mask.any():
                out[col] = s.where(~mask)
    return out

def _kind(series, info):
    """numeric | categorical | boolean | datetime | text for one column."""
    declared = info.get("type")
    if declared in NUMERIC_TYPES:
        return "numeric"
    if declared in ("categorical", "boolean", "datetime", "text"):
        return declared
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"

def _global_mode(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if len(modes) else None

def _group_mode(df, col, keys):
    """Most frequent value of `col` within each group of `keys`, aligned to df's rows."""
    known = df.dropna(subset=[col])
    counts = known.groupby(keys + [col], dropna=False).size().reset_index(name="_n")
    best = (counts.sort_values(["_n"], ascending=False, kind="stable")
                  .drop_duplicates(keys)[keys + [col]]
                  .rename(columns={col: "_fill"}))
    merged = df[keys].merge(best, on=keys, how="left")
    return pd.Series(merged["_fill"].to_numpy(), index=df.index)

def _impute_numeric(out, col, info, icfg, group_keys):
    """Returns (n_filled, detail). Changes out[col] in place."""
    log_detail = {}
    if not pd.api.types.is_numeric_dtype(out[col]):
        before_valid = int(out[col].notna().sum())
        out[col] = pd.to_numeric(out[col], errors="coerce")
        coerced = before_valid - int(out[col].notna().sum())
        if coerced > 0:
            log_detail["coerced_to_missing"] = coerced
    stat = icfg["numeric_strategy"]
    missing_before = int(out[col].isna().sum())
    if missing_before == 0 or out[col].notna().sum() == 0:
        return 0, log_detail, missing_before
    global_value = out[col].mean() if stat == "mean" else out[col].median()
    if info.get("type") == "integer":
        global_value = round(global_value)

    if group_keys:
        by_group = out.groupby(group_keys, dropna=False)[col].transform(stat)
        if info.get("type") == "integer":
            by_group = by_group.round()
        filled = out[col].fillna(by_group)
        n_fallback = int(filled.isna().sum())
        out[col] = filled.fillna(global_value)
        log_detail.update({"strategy": f"group_{stat}", "group_by": list(group_keys),
                           "n_fallback": n_fallback, "fallback_value": _clean(global_value)})
    else:
        out[col] = out[col].fillna(global_value)
        log_detail.update({"strategy": stat, "fill_value": _clean(global_value)})

    if info.get("type") == "integer" and out[col].notna().all():
        try:
            out[col] = out[col].round().astype("int64")
        except (ValueError, TypeError):
            pass
    return missing_before - int(out[col].isna().sum()), log_detail, missing_before

def _impute_categorical(out, col, kind, icfg, group_keys, hide_value):
    strategy = icfg["boolean_strategy"] if kind == "boolean" else icfg["categorical_strategy"]
    if isinstance(out[col].dtype, pd.CategoricalDtype):
        out[col] = out[col].astype(object)
    missing_before = int(out[col].isna().sum())
    constant = icfg["categorical_constant"]
    global_value = constant if strategy == "constant" else _global_mode(out[col])
    used_constant_fallback = global_value is None
    if global_value is None:
        global_value = constant
    detail = {}

    if strategy == "mode" and group_keys:
        by_group = _group_mode(out, col, list(group_keys))
        filled = out[col].fillna(by_group)
        n_fallback = int(filled.isna().sum())
        out[col] = filled.fillna(global_value)
        detail.update({"strategy": "group_mode", "group_by": list(group_keys), "n_fallback": n_fallback})
        if not hide_value:
            detail["fallback_value"] = _clean(global_value)
    else:
        out[col] = out[col].fillna(global_value)
        detail["strategy"] = "constant" if (strategy == "constant" or used_constant_fallback) else "mode"
        if not hide_value:
            detail["fill_value"] = _clean(global_value)
    return missing_before - int(out[col].isna().sum()), detail, missing_before

def _clean(value):
    """Plain Python value, safe to write to JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def impute_statistical(df, schema=None, config=None, columns=None):
    """Fill missing values with simple statistics.

    df      : the dataset (not changed)
    schema  : column types and meaning (Person 1's schema; a basic stand-in is built if None)
    config  : optional dict of settings to change (see ds_config.yaml)
    columns : optional list of columns to work on (default: all)
    Returns (cleaned_df, log) where log is a list of dicts for cleaning_log.json.
    """
    cfg = get_config(config)
    icfg = cfg["imputation"]
    schema = schema or basic_schema(df)
    out = apply_null_markers(df, cfg["null_markers"])
    log = []

    # Values that were only "missing" because they were markers like N/A
    for col in out.columns:
        n_new = int(out[col].isna().sum()) - int(df[col].isna().sum())
        if n_new > 0:
            log.append(_log_entry(col, "marker_to_null", n_changed=n_new))

    n_rows = len(out)
    for col in (columns if columns is not None else list(out.columns)):
        if col not in out.columns:
            log.append(_log_entry(col, "skipped", reason="column not in dataset"))
            continue
        try:
            info = col_info(schema, col)
            n_missing = int(out[col].isna().sum())
            if n_missing == 0:
                continue
            semantic = info.get("semantic")
            kind = _kind(out[col], info)

            reason = None
            if col in icfg["skip_columns"]:
                reason = "column is in skip_columns"
            elif semantic in icfg["skip_semantics"]:
                reason = f"{semantic} column: not imputed"
            elif kind == "datetime":
                reason = "date column: dates are not invented"
            elif kind == "text" and not icfg["impute_text"]:
                reason = "free-text column: not imputed"
            elif n_missing == n_rows:
                reason = "every value is missing"
            elif n_missing / n_rows > icfg["max_missing_share"]:
                reason = (f"{n_missing / n_rows:.0%} missing, above max_missing_share "
                          f"({icfg['max_missing_share']:.0%})")
            if reason:
                log.append(_log_entry(col, "skipped", reason=reason, n_missing=n_missing))
                continue

            group_keys = [k for k in icfg["group_by"].get(col, []) if k in out.columns and k != col]
            if kind == "numeric":
                n_filled, detail, n_before = _impute_numeric(out, col, info, icfg, group_keys)
            else:
                n_filled, detail, n_before = _impute_categorical(
                    out, col, kind, icfg, group_keys, hide_value=semantic in icfg["skip_semantics"])
            if n_filled == 0:
                log.append(_log_entry(col, "skipped", reason="no known values to learn from",
                                      n_missing=n_before, **{k: v for k, v in detail.items()
                                                             if k == "coerced_to_missing"}))
                continue
            log.append(_log_entry(col, "filled_missing", n_changed=n_filled,
                                  n_missing_before=n_before, **detail))
        except Exception as exc:   # one broken column must not stop the others
            log.append(_log_entry(col, "error", error=f"{type(exc).__name__}: {exc}"))
    return out, log

# baseline scoring
def mask_known_values(df, column, share=0.1, seed=0):
    """Hide a random share of the known values in one column, for testing imputers.

    Returns (masked_df, truth, masked_index). Use the same share and seed for every imputer
    (statistical, KNN, regression...) so they are scored on exactly the same hidden values.
    """
    rng = np.random.default_rng(seed)
    known = df.index[df[column].notna()].to_numpy()
    k = max(1, int(len(known) * share)) if len(known) else 0
    masked_index = rng.choice(known, size=k, replace=False) if k else np.array([], dtype=known.dtype)
    masked = df.copy()
    truth = df.loc[masked_index, column].copy()
    masked.loc[masked_index, column] = np.nan
    return masked, truth, masked_index

def evaluate_statistical(df, schema=None, config=None, columns=None,
                         mask_share=0.1, seed=0, max_rows=100_000, min_known=20):
    """How good is the statistical baseline? Hide known values, fill them, compare.

    Returns {column: {"kind": "numeric", "mae": ..., "rmse": ...}
                    or {"kind": "categorical", "accuracy": ...}
                    or {"kind": "skipped", "reason": ...}}
    Lower mae/rmse and higher accuracy are better. ML imputers should beat these numbers.
    """
    cfg = get_config(config)
    schema = schema or basic_schema(df)
    data = apply_null_markers(df, cfg["null_markers"])
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=seed)
    data = data.reset_index(drop=True)

    results = {}
    for col in (columns if columns is not None else list(data.columns)):
        info = col_info(schema, col)
        kind = _kind(data[col], info)
        if kind in ("datetime", "text") or info.get("semantic") in cfg["imputation"]["skip_semantics"]:
            results[col] = {"kind": "skipped", "reason": "column type is not imputed"}
            continue
        if kind == "numeric" and not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].notna().sum() < min_known:
            results[col] = {"kind": "skipped", "reason": f"fewer than {min_known} known values"}
            continue
        masked, truth, idx = mask_known_values(data, col, mask_share, seed)
        filled, _ = impute_statistical(masked, schema, config, columns=[col])
        predicted = filled.loc[idx, col]
        if predicted.isna().any():
            results[col] = {"kind": "skipped", "reason": "imputer left some hidden values empty"}
        elif kind == "numeric":
            err = (predicted.astype(float) - truth.astype(float)).abs()
            results[col] = {"kind": "numeric", "mae": float(err.mean()),
                            "rmse": float(np.sqrt((err ** 2).mean()))}
        else:
            results[col] = {"kind": "categorical",
                            "accuracy": float((predicted.astype(str) == truth.astype(str)).mean())}
    return results