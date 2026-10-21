import pandas as pd
import pytest

import column_rules as cr
import engine


def make_config(**rules):
    """Build a small config in the test, so tests never depend on the real rules.yaml."""
    return {"version": 1, "null_markers": ["", "N/A"], "rules": rules}


def flagged(flags):
    return [f["column"] for f in flags]


# individual rules

def test_all_null_flags_empty_column():
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    assert flagged(cr.check_all_null(df, {"severity": "high"})) == ["a"]


def test_all_null_ignores_empty_dataframe():
    df = pd.DataFrame({"a": []})
    assert cr.check_all_null(df, {"severity": "high"}) == []


def test_constant_flags_single_value_column():
    df = pd.DataFrame({"a": ["x", "x", "x"], "b": [1, 2, 3]})
    assert flagged(cr.check_constant(df, {"severity": "medium"})) == ["a"]


def test_constant_ignores_one_row_dataframe():
    df = pd.DataFrame({"a": ["x"]})
    assert cr.check_constant(df, {"severity": "medium"}) == []


def test_near_unique():
    cfg = {"severity": "low", "threshold": 0.95, "min_rows": 20,
           "id_name_patterns": ["id", "code"]}
    n = 30
    df = pd.DataFrame({
        "note_text": [f"t{i}" for i in range(n)],       # should be flagged
        "county_code": [f"c{i}" for i in range(n)],     # ID-like name: skipped
        "userId": [f"u{i}" for i in range(n)],          # camelCase ID: skipped
        "amount": [i + 0.5 for i in range(n)],          # decimals: skipped
    })
    assert flagged(cr.check_near_unique(df, cfg)) == ["note_text"]


def test_near_unique_skips_small_dataset():
    cfg = {"severity": "low", "threshold": 0.95, "min_rows": 20,
           "id_name_patterns": ["id"]}
    df = pd.DataFrame({"note_text": ["a", "b", "c", "d", "e"]})
    assert cr.check_near_unique(df, cfg) == []


def test_leakage_matches_whole_words_only():
    cfg = {"severity": "medium", "patterns": ["target"]}
    df = pd.DataFrame({"target_value": [1], "targeted": [1], "amount": [1]})
    assert flagged(cr.check_leakage_names(df, cfg)) == ["target_value"]


def test_skew_flags_lopsided_column():
    cfg = {"severity": "low", "threshold": 5.0}
    df = pd.DataFrame({
        "x": [1] * 47 + [2, 3, 1000],     # a few huge values
        "y": list(range(50)),             # balanced
    })
    assert flagged(cr.check_skew(df, cfg)) == ["x"]

def test_skew_skips_identifier_like_columns():
    cfg = {"severity": "low", "threshold": 5.0, "skip_name_patterns": ["county", "code"]}
    lopsided = [1] * 47 + [2, 3, 1000]
    df = pd.DataFrame({"County": lopsided, "state_code": lopsided, "indemnity": lopsided})
    assert [f["column"] for f in cr.check_skew(df, cfg)] == ["indemnity"]


def test_yaml_booleans_in_word_lists_give_a_clear_error():
    import yaml
    config = yaml.safe_load("""
version: 1
rules:
  near_unique:
    severity: low
    threshold: 0.95
    min_rows: 20
    id_name_patterns: [id, no]
""")
    with pytest.raises(ValueError, match="quotes"):
        engine.validate_config(config)

# the engine

ALL_RULES = dict(
    all_null={"severity": "high"},
    constant_column={"severity": "medium"},
    near_unique={"severity": "low", "threshold": 0.95, "min_rows": 20,
                 "id_name_patterns": ["id", "code"]},
    leakage_names={"severity": "medium", "patterns": ["target"]},
    extreme_skew={"severity": "low", "threshold": 5.0},
)


def test_clean_table_gives_no_flags():
    df = pd.DataFrame({
        "state": ["IA", "IL", "NE", "KS", "MO"] * 5,
        "amount": [(i % 7) + 0.5 for i in range(25)],
    })
    result = engine.run_rules(df, config=make_config(**ALL_RULES))
    assert result["flags"] == []
    assert result["errors"] == []


def test_disabled_rule_does_not_run():
    df = pd.DataFrame({"a": [None, None, None]})
    config = make_config(all_null={"enabled": False, "severity": "high"})
    assert engine.run_rules(df, config=config)["flags"] == []


def test_null_markers_count_as_missing():
    df = pd.DataFrame({"a": ["N/A", "N/A", "N/A"], "b": [1, 2, 3]})
    config = make_config(all_null={"severity": "high"})
    result = engine.run_rules(df, config=config)
    assert flagged(result["flags"]) == ["a"]


def test_flag_has_expected_fields():
    df = pd.DataFrame({"a": [None, None, None]})
    config = make_config(all_null={"severity": "high"})
    flag = engine.run_rules(df, config=config)["flags"][0]
    assert set(flag) == {"column", "rule", "severity", "confidence",
                         "evidence", "suggestion"}
    assert 0 <= flag["confidence"] <= 1


def test_most_serious_flags_come_first():
    df = pd.DataFrame({"a": [None, None, None], "b": ["x", "x", "x"]})
    config = make_config(all_null={"severity": "high"},
                         constant_column={"severity": "medium"})
    severities = [f["severity"] for f in engine.run_rules(df, config=config)["flags"]]
    assert severities == ["high", "medium"]


# config checks

@pytest.mark.parametrize("bad_rules", [
    {"all_null": {"severity": "high", "treshold": 1}},    # typo in a setting
    {"made_up_rule": {"severity": "high"}},               # unknown rule
    {"all_null": {"severity": "critical"}},               # bad severity
    {"all_null": {"severity": "high", "confidence": 1.5}},  # out of range
])
def test_bad_config_raises(bad_rules):
    with pytest.raises(ValueError):
        engine.validate_config(make_config(**bad_rules))