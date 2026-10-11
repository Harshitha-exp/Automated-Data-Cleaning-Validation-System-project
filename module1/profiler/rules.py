"""rules.py -- OWNER: Person 3. Suspicious columns, format inconsistency, PII.

Depends on Person 1's Metadata (semantic types). Until that is ready, test with a mocked Metadata.
Thresholds live in rules_config.yaml so they can be tuned without code changes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .schemas import Flag, Metadata, Statistics

CONFIG_PATH = Path(__file__).with_name("rules_config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def detect_suspicious_columns(df: pd.DataFrame, stats: Statistics, cfg: dict) -> list[Flag]:
    """Constant, all-null, near-unique non-ID, leakage-like names, extreme skew.
    TODO(P3): all-null, leakage names, skew threshold, near-unique."""
    flags = []
    for card in stats.cardinality:
        if card.is_constant:
            flags.append(Flag(column=card.column, rule_id="SUSP_CONSTANT", category="suspicious",
                              severity="medium", confidence=1.0,
                              message=f"Column '{card.column}' has <=1 unique value.",
                              evidence={"n_unique": card.n_unique}))
    return flags


def detect_format_inconsistency(df: pd.DataFrame, metadata: Metadata, cfg: dict) -> list[Flag]:
    """Mixed date formats, mixed casing, inconsistent phone patterns, mixed types.
    TODO(P3)."""
    flags = []
    for col in metadata.columns:
        if col.is_mixed_type:
            flags.append(Flag(column=col.name, rule_id="FMT_MIXED_TYPE", category="format_inconsistency",
                              severity="medium", confidence=0.8,
                              message=f"Column '{col.name}' mixes several value types.",
                              evidence={"type_distribution": col.type_distribution}))
    return flags


def detect_pii(df: pd.DataFrame, metadata: Metadata, cfg: dict) -> list[Flag]:
    """Name heuristics + regex + (optional) Presidio. NEVER return raw values in evidence.
    TODO(P3): phone, SSN/national ID, credit card (Luhn), IP, DOB, address, names via NER."""
    flags = []
    for col in metadata.columns:
        if col.semantic_type == "email":
            flags.append(Flag(column=col.name, rule_id="PII_EMAIL", category="pii",
                              severity="high", confidence=col.semantic_confidence,
                              message=f"Column '{col.name}' likely contains email addresses.",
                              evidence={"detected_by": "semantic_type"}))
    return flags


def run_rules(df: pd.DataFrame, metadata: Metadata, stats: Statistics, cfg: dict | None = None) -> list[Flag]:
    cfg = cfg or load_config()
    return (detect_suspicious_columns(df, stats, cfg)
            + detect_format_inconsistency(df, metadata, cfg)
            + detect_pii(df, metadata, cfg))
