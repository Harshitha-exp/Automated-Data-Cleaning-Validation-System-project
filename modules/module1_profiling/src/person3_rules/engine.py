from pathlib import Path
import numpy as np
import pandas as pd
import yaml                      # reads the rules.yaml settings file
import column_rules as cr        # the column checks (constant, null, skew...)
import formats                   # inconsistent formats
import pii                       # personal data (PII)


# Connects each rule name in rules.yaml to the function that runs it.
# To add a new rule later, write its function and add one line here.
CHECKS = {
    "all_null": cr.check_all_null,
    "constant_column": cr.check_constant,
    "near_unique": cr.check_near_unique,
    "leakage_names": cr.check_leakage_names,
    "extreme_skew": cr.check_skew,
    "format_inconsistency": formats.detect_format_inconsistency,
    "pii": pii.detect_pii,
}

# Used to sort flags so the most serious problems appear first
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Every setting name a rule is allowed to have (anything else is probably a typo)
ALLOWED_KEYS = {
    "enabled", "severity", "confidence", "threshold", "min_rows",
    "id_name_patterns", "patterns", "skip_name_patterns",
    # format_inconsistency
    "dominant_share", "date_share", "phone_share", "max_shapes",
    "max_distinct_ratio", "sample_size",
    # pii
    "min_confidence", "regex_min_share", "use_presidio", "presidio_model",
    "presidio_sample", "presidio_min_share", "presidio_min_score",
    "weights", "name_hint_required",
}

SUPPORTED_VERSION = 1


def validate_config(config):
    """Stop early, with a clear message, if rules.yaml has a mistake."""
    if not isinstance(config, dict) or not isinstance(config.get("rules"), dict):
        raise ValueError("rules.yaml must contain a 'rules:' section")

    if config.get("version", SUPPORTED_VERSION) != SUPPORTED_VERSION:
        raise ValueError(f"unsupported config version: {config.get('version')}")

    for name, cfg in config["rules"].items():
        if name not in CHECKS:
            raise ValueError(f"unknown rule '{name}' in rules.yaml")
        cfg = cfg or {}
        unknown = set(cfg) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"rule '{name}' has unknown setting(s): {sorted(unknown)}")
        if cfg.get("severity") not in SEVERITY_ORDER:
            raise ValueError(f"rule '{name}' needs severity: high, medium or low")
        for key in ("id_name_patterns", "patterns", "skip_name_patterns", "name_hint_required"):
            for item in cfg.get(key, []):
                if not isinstance(item, str):
                    raise ValueError(
                        f"rule '{name}' setting '{key}' has a non-text item ({item!r}). "
                        f"In YAML, put quotes around words like no, yes, on, off, y, n.")
        confidence = cfg.get("confidence", 1.0)
        if not 0 <= confidence <= 1:
            raise ValueError(f"rule '{name}' confidence must be between 0 and 1")


def load_config(path=None):
    """Read and check the settings from rules.yaml."""
    path = path or Path(__file__).parent / "rules.yaml"
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config)
    return config


def apply_null_markers(df, markers):
    """Return a copy of df where marker values (like 'N/A' or -999) become real nulls."""
    if not markers:
        return df
    to_replace = list(markers)
    for m in markers:               # also catch the numeric version, e.g. -999
        try:
            to_replace.append(float(m))
        except ValueError:
            pass
    return df.replace(to_replace, np.nan)


def run_rules(df, metadata=None, config=None):
    """Run every enabled rule on the dataset and return the flags.

    df       : the dataset (a pandas DataFrame)
    metadata : Person 1's column information (optional, rules work without it)
    config   : settings to use (optional, loads rules.yaml if not given)

    Returns a dictionary with the flags, a total count, and any rules that failed.
    This is what goes into the 'flags' section of profiling_report.json.
    """
    if config is None:
        config = load_config()
    else:
        validate_config(config)

    df = apply_null_markers(df, config.get("null_markers", []))

    flags = []
    errors = []

    for name, check in CHECKS.items():
        cfg = config["rules"].get(name)
        if cfg is None:                          # rule not listed in the YAML: skip it
            continue
        if not cfg.get("enabled", True):         # switched off
            continue
        try:
            flags.extend(check(df, cfg, metadata))
        except Exception as e:                   # one broken rule shouldn't stop the others
            errors.append({"rule": name, "error": f"{type(e).__name__}: {e}"})

    # Most serious first; within the same severity, most confident first
    flags.sort(key=lambda f: (SEVERITY_ORDER[f["severity"]], -f["confidence"]))
    return {"flags": flags, "total_flags": len(flags), "errors": errors}