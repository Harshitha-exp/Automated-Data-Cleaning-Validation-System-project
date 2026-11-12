"""Settings loader for Person 2's part of Module 2 (statistical imputation and quality scoring).

Settings live in ds_config.yaml. A misspelt or invalid setting raises a clear
error when the settings are loaded, the same way the rule engine does.
"""
import copy
from pathlib import Path
import yaml

DEFAULT_PATH = Path(__file__).parent / "ds_config.yaml"
SUPPORTED_VERSION = 1

ALLOWED_KEYS = {
    "top": {"version", "null_markers", "imputation", "quality"},
    "imputation": {"max_missing_share", "numeric_strategy", "categorical_strategy",
                   "categorical_constant", "boolean_strategy", "impute_text",
                   "skip_semantics", "skip_columns", "group_by"},
    "quality": {"weights", "sample_size", "duplicate_subset", "categorical_max_ratio"},
}
NUMERIC_STRATEGIES = {"mean", "median"}
CATEGORICAL_STRATEGIES = {"mode", "constant"}
QUALITY_DIMENSIONS = {"completeness", "uniqueness", "validity", "consistency"}

def _merge(base, override):
    """Copy of base with the values from override laid on top (sections are merged)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "group_by":
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out

def _check_text_list(name, value):
    if not isinstance(value, list):
        raise ValueError(f"'{name}' must be a list")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"'{name}' has a non-text item ({item!r}). "
                             f"In YAML, put quotes around words like no, yes, on, off, y, n.")

def validate_ds_config(config):
    """Stop early, with a clear message, if the settings have a mistake."""
    if not isinstance(config, dict):
        raise ValueError("settings must be a mapping")
    if config.get("version", SUPPORTED_VERSION) != SUPPORTED_VERSION:
        raise ValueError(f"unsupported config version: {config.get('version')}")

    unknown = set(config) - ALLOWED_KEYS["top"]
    if unknown:
        raise ValueError(f"unknown setting(s): {sorted(unknown)}")
    for section in ("imputation", "quality"):
        unknown = set(config.get(section, {})) - ALLOWED_KEYS[section]
        if unknown:
            raise ValueError(f"'{section}' has unknown setting(s): {sorted(unknown)}")

    _check_text_list("null_markers", config.get("null_markers", []))

    imp = config.get("imputation", {})
    if not 0 < imp.get("max_missing_share", 0.6) <= 1:
        raise ValueError("'max_missing_share' must be above 0 and at most 1")
    if imp.get("numeric_strategy", "median") not in NUMERIC_STRATEGIES:
        raise ValueError(f"'numeric_strategy' must be one of {sorted(NUMERIC_STRATEGIES)}")
    for key in ("categorical_strategy", "boolean_strategy"):
        if imp.get(key, "mode") not in CATEGORICAL_STRATEGIES:
            raise ValueError(f"'{key}' must be one of {sorted(CATEGORICAL_STRATEGIES)}")
    _check_text_list("skip_semantics", imp.get("skip_semantics", []))
    _check_text_list("skip_columns", imp.get("skip_columns", []))
    group_by = imp.get("group_by", {})
    if not isinstance(group_by, dict):
        raise ValueError("'group_by' must be a mapping of column -> list of columns")
    for col, keys in group_by.items():
        if not isinstance(col, str):
            raise ValueError(f"'group_by' column name {col!r} must be text")
        _check_text_list(f"group_by[{col}]", keys)

    qual = config.get("quality", {})
    weights = qual.get("weights", {})
    if set(weights) - QUALITY_DIMENSIONS:
        raise ValueError(f"'weights' has unknown dimension(s): {sorted(set(weights) - QUALITY_DIMENSIONS)}")
    if any((not isinstance(w, (int, float))) or w < 0 for w in weights.values()):
        raise ValueError("'weights' must be numbers that are 0 or more")
    if weights and sum(weights.values()) <= 0:
        raise ValueError("'weights' must not all be zero")
    if qual.get("sample_size", 1) < 1:
        raise ValueError("'sample_size' must be at least 1")
    subset = qual.get("duplicate_subset")
    if subset is not None:
        _check_text_list("duplicate_subset", subset)
    if not 0 < qual.get("categorical_max_ratio", 0.5) <= 1:
        raise ValueError("'categorical_max_ratio' must be above 0 and at most 1")

def get_config(config=None, path=None):
    """Settings from ds_config.yaml, with any values in `config` laid on top.

    config : optional dict with only the settings you want to change
    path   : optional other YAML file to use instead of ds_config.yaml
    """
    with open(path or DEFAULT_PATH, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    merged = _merge(base, config)
    validate_ds_config(merged)
    return merged