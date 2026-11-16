import numpy as np
import pandas as pd
import pytest
import impute_stats as ist
import quality_score as qs
from ds_config import get_config

def tidy():
    return pd.DataFrame({
        "crop": ["Corn", "Wheat", "Oats", "Corn"] * 5,
        "amount": [float(i) for i in range(20)],
    })

def test_a_tidy_table_scores_full_marks():
    score = qs.score_quality(tidy())
    assert score["dimensions"]["completeness"] == 1.0
    assert score["dimensions"]["uniqueness"] == 1.0
    assert score["dimensions"]["consistency"] == 1.0
    assert score["overall"] is not None

def test_completeness_counts_missing_cells():
    df = tidy()
    df.loc[0, "amount"] = np.nan
    df.loc[1, "amount"] = np.nan
    score = qs.score_quality(df)
    assert score["dimensions"]["completeness"] == pytest.approx(1 - 2 / 40, abs=1e-4)
    assert score["n_missing_cells"] == 2

def test_marker_values_count_as_missing():
    df = tidy()
    df.loc[0, "crop"] = "N/A"
    assert qs.score_quality(df)["n_missing_cells"] == 1

def test_duplicates_lower_uniqueness():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert qs.score_quality(df)["dimensions"]["uniqueness"] == 1.0
    assert qs.score_quality(dup)["dimensions"]["uniqueness"] == pytest.approx(0.8)
    assert qs.score_quality(dup)["n_duplicate_rows"] == 1

def test_duplicate_subset_can_be_chosen():
    df = pd.DataFrame({"id": [1, 1, 2], "note": ["a", "b", "c"]})
    score = qs.score_quality(df, config={"quality": {"duplicate_subset": ["id"]}})
    assert score["n_duplicate_rows"] == 1

def test_validity_uses_ranges_from_the_schema():
    df = pd.DataFrame({"payout": [10.0, 20.0, -5.0, 30.0]})
    schema = {"columns": {"payout": {"type": "float", "min": 0}}}
    assert qs.score_quality(df, schema)["dimensions"]["validity"] == pytest.approx(0.75)

def test_validity_catches_text_in_number_columns():
    df = pd.DataFrame({"x": ["1", "2", "abc", "4"]})
    schema = {"columns": {"x": {"type": "float"}}}
    assert qs.score_quality(df, schema)["dimensions"]["validity"] == pytest.approx(0.75)

def test_validity_checks_allowed_values_and_formats():
    df = pd.DataFrame({"crop": ["Corn", "Wheat", "Rice", "Corn"], "fips": ["01001", "1001", "19153", "17031"]})
    schema = {"columns": {"crop": {"type": "categorical", "allowed_values": ["Corn", "Wheat"]},
                          "fips": {"type": "categorical", "format_regex": r"\d{5}"}}}
    cols = qs.score_quality(df, schema)["columns"]
    assert cols["crop"]["validity"] == pytest.approx(0.75)
    assert cols["fips"]["validity"] == pytest.approx(0.75)

def test_validity_is_not_scored_when_there_are_no_rules():
    score = qs.score_quality(tidy())
    assert score["dimensions"]["validity"] is None
    assert any("validity not scored" in n for n in score["notes"])

def test_the_overall_score_ignores_missing_dimensions_instead_of_counting_them_as_perfect():
    score = qs.score_quality(tidy())
    assert "validity" not in score["weights_used"]
    assert sum(score["weights_used"].values()) == pytest.approx(1.0)

def test_consistency_catches_casing_variants_and_stray_spaces():
    df = pd.DataFrame({"crop": ["Corn"] * 8 + ["CORN", " Wheat"] + ["Wheat"] * 8})
    score = qs.score_quality(df)
    assert score["dimensions"]["consistency"] < 1.0
    assert qs.score_quality(df.assign(crop=df["crop"].str.strip().str.title()))["dimensions"]["consistency"] == 1.0

def test_changing_the_weights_changes_the_overall_score():
    df = tidy()
    df.loc[0, "amount"] = np.nan
    a = qs.score_quality(df, config={"quality": {"weights": {"completeness": 1.0, "uniqueness": 0.0, "consistency": 0.0}}})
    b = qs.score_quality(df, config={"quality": {"weights": {"completeness": 0.0, "uniqueness": 1.0, "consistency": 0.0}}})
    assert a["overall"] < b["overall"]

def test_an_empty_table_does_not_crash():
    score = qs.score_quality(pd.DataFrame({"a": []}))
    assert score["overall"] is None and score["notes"]

def test_large_tables_are_sampled_for_validity_and_consistency():
    df = pd.DataFrame({"x": np.arange(300, dtype=float)})
    score = qs.score_quality(df, config={"quality": {"sample_size": 100}})
    assert score["sampled_rows"] == 100 and score["n_rows"] == 300

def test_delta_is_after_minus_before():
    before = {"overall": 80.0, "dimensions": {"completeness": 0.8, "uniqueness": 1.0, "validity": None, "consistency": 0.9}}
    after = {"overall": 90.0, "dimensions": {"completeness": 1.0, "uniqueness": 1.0, "validity": None, "consistency": 0.85}}
    delta = qs.quality_delta(before, after)
    assert delta["overall_delta"] == 10.0
    assert delta["improved"] == ["completeness"] and delta["worsened"] == ["consistency"]
    assert "validity" not in delta["dimensions"]

def test_cleaning_then_scoring_shows_an_improvement():
    raw = pd.DataFrame({
        "crop": ["Corn", "Corn", "Wheat", None, "Corn", "N/A"] * 4,
        "amount": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0] * 4,
    })
    cleaned, log = ist.impute_statistical(raw)
    result = qs.score_before_after(raw, cleaned)
    assert result["delta"]["overall_delta"] > 0
    assert result["after"]["dimensions"]["completeness"] == 1.0
    assert result["before"]["n_missing_cells"] > result["after"]["n_missing_cells"]
    assert log

def test_bad_quality_settings_give_a_clear_error():
    with pytest.raises(ValueError):
        get_config({"quality": {"weights": {"speed": 1.0}}})
    with pytest.raises(ValueError):
        get_config({"quality": {"weights": {"completeness": -1}}})
    with pytest.raises(ValueError):
        get_config({"qualitty": {}})

def test_yaml_booleans_in_word_lists_give_a_clear_error():
    import yaml
    config = yaml.safe_load("imputation:\n  skip_columns: [id, no]\n")
    with pytest.raises(ValueError, match="quotes"):
        get_config(config)

def test_the_shipped_settings_file_is_valid():
    cfg = get_config()
    assert cfg["version"] == 1 and "null_markers" in cfg