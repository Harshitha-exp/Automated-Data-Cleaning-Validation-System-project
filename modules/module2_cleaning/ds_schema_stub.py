"""STAND-IN schema for Person 2's code, until the real schema inference engine exists.

The real schema is Person 1's job (schema inference). Until it is ready, this builds a
minimal one from the dataframe so imputation and scoring can run and be tested. The
shape follows Module 1's metadata, so the real schema should be a drop-in:

    {"columns": {col: {"type": "integer|float|boolean|datetime|categorical|text",
                       "semantic": "id|email|...|None",
                       "min": number or None, "max": number or None,
                       "allowed_values": list or None,
                       "format_regex": text or None,
                       "nullable": True/False}}}

Only "type" and "semantic" are needed for imputation. Validity scoring also uses min, max,
allowed_values and format_regex when they are present. This stub never invents ranges
from the data (that would make every value look valid): pass overrides for those.
"""
import re
import pandas as pd

ID_WORDS = {"id", "uuid", "key", "code", "fips", "number"}

def _tokens(name):
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {w for w in re.split(r"[\s_\-]+", text.lower()) if w}

def _guess_type(series, categorical_max_unique=50, categorical_max_ratio=0.5):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"
    n_unique = non_null.nunique()
    if n_unique <= categorical_max_unique or n_unique / len(non_null) <= categorical_max_ratio:
        return "categorical"
    return "text"

def basic_schema(df, overrides=None):
    """Minimal schema from a dataframe. overrides: {column: {key: value}} laid on top."""
    columns = {}
    for col in df.columns:
        columns[col] = {
            "type": _guess_type(df[col]),
            "semantic": "id" if _tokens(col) & ID_WORDS else None,
            "min": None, "max": None, "allowed_values": None, "format_regex": None,
            "nullable": True,
        }
    for col, extra in (overrides or {}).items():
        if col in columns:
            columns[col].update(extra)
    return {"schema_version": "stub-0", "columns": columns}

def col_info(schema, col):
    """The schema entry for one column ({} if there is none)."""
    return ((schema or {}).get("columns") or {}).get(col) or {}