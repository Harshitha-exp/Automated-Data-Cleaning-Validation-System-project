import re
import pandas as pd

def make_flag(column, rule, severity, confidence, evidence, suggestion=""):
    """Build one flag (a problem report) in the standard format.

    column     : name of the column with the problem
    rule       : which rule found it
    severity   : how serious it is
    confidence : how sure we are, from 0 to 1
    evidence   : the facts that triggered the flag (so a person can verify it)
    suggestion : what the user could do about it
    """
    return{
        "column": column,
        "rule": rule,
        "severity": severity,
        "confidence": round(min(max(confidence, 0.0), 1.0), 2),
        "evidence": evidence,
        "suggestion": suggestion,
    }

def get_semantic(metadata, col):
    """Look up Person 1's label for a column (e.g. 'id', 'email')."""
    if not metadata:
        return None
    label = metadata.get("columns", {}).get(col, {}).get("semantic")
    return str(label).lower() if label is not None else None

def name_words(col):
    """Split a column name into lowercase words: 'countyCode' -> {'county', 'code'}."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col))
    return {w for w in re.split(r"[\s_\-]+", text.lower()) if w}

def check_all_null(df, cfg, metadata=None):
    """Flag columns where every value is missing."""
    flags = []
    for col in df.columns:
        # len(df) > 0 stops an empty dataset from flagging every column
        if len(df) > 0 and df[col].isna().all():
            flags.append(make_flag(col, "all_null", cfg["severity"], 1.0,
                                   "100% of values are null", "drop this column"))
    return flags

def check_constant(df, cfg, metadata=None):
    """Flag columns that contain only one distinct value."""
    flags = []

    # With fewer than 2 rows, every column looks constant, so skip
    if len(df) < 2:
        return flags

    for col in df.columns:
        # Count distinct values, ignoring empty cells
        if df[col].nunique(dropna=True) == 1:
            # show the repeated value as evidence
            value = df[col].dropna().iloc[0]
            n_null = int(df[col].isna().sum())
            shown = repr(value) if pd.api.types.is_numeric_dtype(df[col]) else "<hidden>"
            flags.append(make_flag(col, "constant_column", cfg["severity"], 1.0,
                                   f"only one distinct value: {shown} ({n_null} nulls)",
                                   "likely useless for analysis, consider dropping"))
    return flags

def check_near_unique(df, cfg, metadata=None):
    """Flag columns where almost every value is different, but which are not IDs.

    IDs are supposed to be unique, so they are skipped. Other columns that are
    almost fully unique may be hidden IDs or free text, which are worth a look.
    """
    flags = []

    # Tiny datasets make every column look unique, so skip them
    if len(df) < cfg["min_rows"]:
        return flags

    # Lower-case set of ID-like words (e.g. "id", "code", "fips")
    id_words = {w.lower() for w in cfg["id_name_patterns"]}

    for col in df.columns:
        # Decimal columns (like money) are naturally almost all different, so skip
        if pd.api.types.is_float_dtype(df[col]):
            continue

        # Skip columns that Person 1 labelled as an ID, or whose name has an ID-like word
        if get_semantic(metadata, col) == "id" or (name_words(col) & id_words):
            continue

        # ignore empty cells
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Share of values that are distinct: 1.0 means every value is different
        ratio = series.nunique() / len(series)
        if ratio > cfg["threshold"]:
            flags.append(make_flag(col, "near_unique", cfg["severity"],
                                   cfg.get("confidence", 0.8),
                                   f"{ratio:.1%} of values are distinct",
                                   "check if this is an undeclared ID or free text"))
    return flags

def check_leakage_names(df, cfg, metadata=None):
    """Flag column names that hint at 'data leakage'.

    Leakage means a column contains information that would not be known in real
    life at prediction time. We can only guess from the name, so confidence is lower.
    """
    flags = []
    patterns = {p.lower() for p in cfg["patterns"]}

    for col in df.columns:
        # Whole-word match, so "target" matches "target_col" but not "targeted"
        hits = sorted(name_words(col) & patterns)
        if hits:
            flags.append(make_flag(col, "leakage_names", cfg["severity"],
                                   cfg.get("confidence", 0.6),
                                   f"column name contains {hits}",
                                   "check this column isn't derived from the prediction target"))
    return flags

def check_skew(df, cfg, metadata=None):
    """Flag number columns where a few huge values dominate.
    Identifier-like columns (codes, IDs, years) are skipped: skew means nothing for them.
    """
    flags = []
    skip_words = {str(w).lower() for w in cfg.get("skip_name_patterns", [])}
    for col in df.columns:
        # Only look at real number columns (True/False columns don't count)
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue

        # Skip columns Person 1 labelled as an ID, or whose name has an ID-like word
        if get_semantic(metadata, col) == "id" or (name_words(col) & skip_words):
            continue

        series = df[col].dropna()
        # skew is meaningless with fewer than 3 distinct values
        if series.nunique() < 3:    
            continue

        skew = series.skew()
        # abs() catches both directions
        if abs(skew) > cfg["threshold"]:    
            flags.append(make_flag(col, "extreme_skew", cfg["severity"],
                                   cfg.get("confidence", 0.9),
                                   f"skew = {skew:.1f} (threshold {cfg['threshold']})",
                                   "consider log transform or winsorising"))
    return flags  
