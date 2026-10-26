"""Inconsistent-format detection (Person 3).

Reports columns where the values do not all follow the same pattern:
mixed date formats, mixed casing, inconsistent phone patterns, and (as a
catch-all) a column where almost every value has one shape but a few differ.

This file only REPORTS problems. It never changes the data.
Evidence strings show abstract patterns (like '9999-99-99'), never real values.
"""
import re
from collections import Counter

import pandas as pd

from column_rules import make_flag, name_words

RULE_NAME = "format_inconsistency"

# Known date layouts, checked in this order (the first match wins).
# NOTE: 'NN/NN/YYYY' cannot tell day-first from month-first by shape alone.
DATE_PATTERNS = {
    "YYYY-MM-DD": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    "YYYY-M-D": re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"),
    "YYYY/MM/DD": re.compile(r"^\d{4}/\d{1,2}/\d{1,2}$"),
    "ISO datetime": re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?"),
    "NN/NN/YYYY": re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    "NN-NN-YYYY": re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$"),
    "NN.NN.YYYY": re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}$"),
    "D Mon YYYY": re.compile(r"^\d{1,2} [A-Za-z]{3,9},? \d{4}$"),
    "Mon D, YYYY": re.compile(r"^[A-Za-z]{3,9}\.? \d{1,2},? \d{4}$"),
}

PHONE_NAME_WORDS = {"phone", "mobile", "tel", "telephone", "fax", "cell"}
PHONE_CHARS = re.compile(r"^\+?[\d\s().\-]{7,20}$")
PLAIN_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
SSN_LIKE = re.compile(r"^\d{3}-\d{2}-\d{4}$")


def _get_cfg(cfg):
    """Use the given settings, or read this rule's section of rules.yaml."""
    if cfg is not None:
        return cfg
    from engine import load_config   # imported here to avoid a circular import
    return load_config()["rules"].get(RULE_NAME, {})


def date_format_of(value):
    """Return the name of the date layout a value matches, or None."""
    for name, pattern in DATE_PATTERNS.items():
        if pattern.match(value):
            return name
    return None


def looks_like_phone(value):
    """True for values shaped like a phone number (formatted, 7 to 15 digits)."""
    if PLAIN_NUMBER.match(value) or SSN_LIKE.match(value) or date_format_of(value):
        return False
    if not PHONE_CHARS.match(value):
        return False
    if not re.search(r"[\s()+\-.]", value):     # digits only: could be any ID
        return False
    return 7 <= sum(ch.isdigit() for ch in value) <= 15


def value_shape(value, max_len=40):
    """Describe a value's shape: digits -> 9, letters -> A, symbols kept."""
    shape = "".join("9" if ch.isdigit() else "A" if ch.isalpha() else ch for ch in value)
    return shape if len(shape) <= max_len else shape[:max_len] + "..."


def _describe(counter, total, top=3):
    """'YYYY-MM-DD (95.0%), NN/NN/YYYY (5.0%)' from a Counter of formats."""
    parts = [f"{name} ({n / total:.1%})" for name, n in counter.most_common(top)]
    if len(counter) > top:
        parts.append(f"+{len(counter) - top} more")
    return ", ".join(parts)


def _text_values(series, sample_size):
    """Non-empty values of one column as stripped text, sampled if very large."""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=0)
    return s


def detect_format_inconsistency(df, cfg=None, metadata=None):
    """Find columns whose values do not all follow one format.

    df       : the dataset
    cfg      : this rule's settings (optional; read from rules.yaml if missing)
    metadata : accepted so every rule has the same signature (not used here)
    Returns a list of flags in the standard format.
    """
    cfg = _get_cfg(cfg)
    severity = cfg.get("severity", "medium")
    confidence = cfg.get("confidence", 0.85)
    min_rows = cfg.get("min_rows", 20)
    dominant_share = cfg.get("dominant_share", 0.95)
    date_share = cfg.get("date_share", 0.8)
    phone_share = cfg.get("phone_share", 0.8)
    max_shapes = cfg.get("max_shapes", 5)
    max_distinct_ratio = cfg.get("max_distinct_ratio", 0.5)
    sample_size = cfg.get("sample_size", 50000)

    flags = []
    for col in df.columns:
        # Numbers and real datetimes are already consistent by type
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        values = _text_values(df[col], sample_size)
        n = len(values)
        if n < min_rows:
            continue
        counts = values.value_counts()          # work on distinct values, not every row

        # 1. mixed date formats
        date_counts = Counter()
        for value, count in counts.items():
            date_counts[date_format_of(value) or "unrecognised"] += int(count)
        matched = n - date_counts.get("unrecognised", 0)
        if matched / n >= date_share:           # this is a date column
            if len(date_counts) > 1:
                flags.append(make_flag(
                    col, RULE_NAME, severity, confidence,
                    f"mixed date formats: {_describe(date_counts, n)}",
                    "convert every value to one format, e.g. YYYY-MM-DD"))
            continue                            # dates need no other format check

        # 2. inconsistent phone patterns
        phone_counts = Counter()
        phone_like = 0
        for value, count in counts.items():
            if looks_like_phone(value):
                phone_like += int(count)
                phone_counts[re.sub(r"\d", "9", value)] += int(count)
        has_phone_name = bool(name_words(col) & PHONE_NAME_WORDS)
        if phone_like / n >= phone_share or (has_phone_name and phone_like / n >= 0.5):
            if len(phone_counts) > 1:
                flags.append(make_flag(
                    col, RULE_NAME, severity, confidence,
                    f"inconsistent phone patterns: {_describe(phone_counts, phone_like)}",
                    "standardise phone numbers to one pattern, e.g. +CC NNNNNNNNNN"))
            continue

        # 3. mixed casing (only for category-like columns)
        if len(counts) / n <= max_distinct_ratio:
            groups = {}
            for value in counts.index:
                if any(ch.isalpha() for ch in value):
                    groups.setdefault(value.lower(), []).append(value)
            mixed = [v for v in groups.values() if len(v) > 1]
            if mixed:
                affected = sum(int(counts[v]) for group in mixed for v in group)
                flags.append(make_flag(
                    col, RULE_NAME, severity, confidence,
                    f"mixed casing: {len(mixed)} value(s) appear in more than one "
                    f"casing, affecting {affected / n:.1%} of rows",
                    "standardise to one casing (e.g. lower case or Title Case)"))

        # 4. one shape dominates but a few others exist
        shapes = Counter()
        for value, count in counts.items():
            shapes[value_shape(value)] += int(count)
        top_shape, top_count = shapes.most_common(1)[0]
        if len(shapes) > 1 and len(shapes) <= max_shapes and top_count / n >= dominant_share:
            flags.append(make_flag(
                col, RULE_NAME, severity, confidence * 0.8,
                f"{top_count / n:.1%} of values look like {top_shape!r}, "
                f"but others differ: {_describe(shapes, n)}",
                "check the odd-looking values (e.g. lost leading zeros, different length)"))
    return flags