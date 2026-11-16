import numpy as np
import pandas as pd
import pytest
import impute_stats as ist
from ds_schema_stub import basic_schema

def entries(log, column=None, action=None):
    return [e for e in log
            if (column is None or e["column"] == column) and (action is None or e["action"] == action)]

def test_numeric_gaps_are_filled_with_the_median():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 100.0, np.nan]})
    out, log = ist.impute_statistical(df)
    assert out["amount"].tolist() == [1.0, 2.0, 3.0, 100.0, 2.5]
    entry = entries(log, "amount", "filled_missing")[0]
    assert entry["n_changed"] == 1 and entry["strategy"] == "median" and entry["fill_value"] == 2.5

def test_mean_strategy_can_be_chosen():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 6.0, np.nan]})
    out, _ = ist.impute_statistical(df, config={"imputation": {"numeric_strategy": "mean"}})
    assert out["amount"].iloc[4] == 3.0

def test_the_input_dataframe_is_never_changed():
    df = pd.DataFrame({"amount": [1.0, np.nan, 3.0], "crop": ["Corn", None, "Corn"]})
    snapshot = df.copy()
    ist.impute_statistical(df)
    pd.testing.assert_frame_equal(df, snapshot)

def test_categories_are_filled_with_the_most_frequent_value():
    df = pd.DataFrame({"crop": ["Corn", "Corn", "Wheat", None]})
    out, log = ist.impute_statistical(df)
    assert out["crop"].tolist() == ["Corn", "Corn", "Wheat", "Corn"]
    assert entries(log, "crop", "filled_missing")[0]["strategy"] == "mode"

def test_constant_strategy_uses_the_chosen_label():
    df = pd.DataFrame({"crop": ["Corn", "Wheat", None]})
    out, _ = ist.impute_statistical(df, config={"imputation": {"categorical_strategy": "constant"}})
    assert out["crop"].iloc[2] == "Unknown"

def test_group_based_fill_uses_similar_rows():
    df = pd.DataFrame({
        "state": [1, 1, 1, 2, 2, 2],
        "county_name": ["Autauga", "Autauga", None, "Benton", "Benton", None],
    })
    config = {"imputation": {"group_by": {"county_name": ["state"]}}}
    out, log = ist.impute_statistical(df, config=config)
    assert out["county_name"].tolist() == ["Autauga", "Autauga", "Autauga", "Benton", "Benton", "Benton"]
    assert entries(log, "county_name", "filled_missing")[0]["strategy"] == "group_mode"

def test_group_with_no_known_value_falls_back_to_the_overall_value():
    df = pd.DataFrame({
        "county": [1, 1, 999, 999],
        "name": ["Autauga", "Autauga", None, None],
    })
    config = {"imputation": {"group_by": {"name": ["county"]}}}
    out, log = ist.impute_statistical(df, config=config)
    assert out["name"].tolist() == ["Autauga"] * 4
    assert entries(log, "name", "filled_missing")[0]["n_fallback"] == 2

def test_group_with_no_known_value_can_fall_back_to_a_constant_instead():
    df = pd.DataFrame({"county": [1, 1, 999, 999], "name": ["Autauga", "Autauga", None, None]})
    config = {"imputation": {"group_by": {"name": ["county"]}, "group_fallback": "constant",
                             "categorical_constant": "Unknown"}}
    out, log = ist.impute_statistical(df, config=config)
    assert out["name"].tolist() == ["Autauga", "Autauga", "Unknown", "Unknown"]
    entry = entries(log, "name", "filled_missing")[0]
    assert entry["n_fallback"] == 2 and entry["fallback_value"] == "Unknown"

def test_bad_group_fallback_gives_a_clear_error():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError):
        ist.impute_statistical(df, config={"imputation": {"group_fallback": "magic"}})

def test_numeric_group_fill():
    df = pd.DataFrame({"g": ["a", "a", "a", "b", "b", "b"],
                       "x": [1.0, 3.0, np.nan, 10.0, 30.0, np.nan]})
    out, _ = ist.impute_statistical(df, config={"imputation": {"group_by": {"x": ["g"]}}})
    assert out["x"].tolist() == [1.0, 3.0, 2.0, 10.0, 30.0, 20.0]

def test_marker_values_count_as_missing_and_are_logged():
    df = pd.DataFrame({"crop": ["Corn", "N/A", "Corn", " NA "], "n": [1.0, 2.0, 3.0, 4.0]})
    out, log = ist.impute_statistical(df)
    assert out["crop"].tolist() == ["Corn"] * 4
    assert entries(log, "crop", "marker_to_null")[0]["n_changed"] == 2

def test_numeric_markers_like_minus_999_are_handled():
    df = pd.DataFrame({"rain": [1.0, -999.0, 3.0]})
    out, _ = ist.impute_statistical(df, config={"null_markers": ["", "-999"]})
    assert out["rain"].tolist() == [1.0, 2.0, 3.0]

def test_id_and_personal_columns_are_skipped():
    df = pd.DataFrame({"farmer_id": ["a", None, "c"], "email": ["x@y.com", None, "z@y.com"]})
    schema = basic_schema(df, overrides={"email": {"semantic": "email"}})
    out, log = ist.impute_statistical(df, schema)
    assert out["farmer_id"].isna().sum() == 1 and out["email"].isna().sum() == 1
    assert len(entries(log, "farmer_id", "skipped")) == 1
    assert len(entries(log, "email", "skipped")) == 1

def test_dates_are_not_invented():
    df = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", None, "2024-01-03"])})
    out, log = ist.impute_statistical(df)
    assert out["when"].isna().sum() == 1
    assert "date" in entries(log, "when", "skipped")[0]["reason"]

def test_free_text_is_not_imputed_by_default():
    df = pd.DataFrame({"note": [f"free text number {i}" for i in range(60)] + [None]})
    out, log = ist.impute_statistical(df)
    assert out["note"].isna().sum() == 1
    assert "free-text" in entries(log, "note", "skipped")[0]["reason"]

def test_columns_with_too_much_missing_are_left_alone():
    df = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, np.nan]})
    out, log = ist.impute_statistical(df)
    assert out["x"].isna().sum() == 4
    assert "max_missing_share" in entries(log, "x", "skipped")[0]["reason"]

def test_all_missing_column_is_skipped_without_crashing():
    df = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, np.nan]})
    out, log = ist.impute_statistical(df, config={"imputation": {"max_missing_share": 1.0}})
    assert out["x"].isna().all() and out["y"].notna().all()
    assert entries(log, "x", "skipped")

def test_integer_columns_stay_integers():
    df = pd.DataFrame({"year": [2001.0, 2002.0, 2003.0, np.nan]})
    schema = {"columns": {"year": {"type": "integer", "semantic": None}}}
    out, _ = ist.impute_statistical(df, schema)
    assert str(out["year"].dtype) == "int64"
    assert out["year"].tolist() == [2001, 2002, 2003, 2002]

def test_text_that_should_be_numeric_is_coerced_and_logged():
    df = pd.DataFrame({"x": ["1", "2", "oops", "4", None]})
    schema = {"columns": {"x": {"type": "float", "semantic": None}}}
    out, log = ist.impute_statistical(df, schema)
    assert out["x"].notna().all()
    entry = entries(log, "x", "filled_missing")[0]
    assert entry["coerced_to_missing"] == 1 and entry["n_changed"] == 2

def test_only_the_requested_columns_are_touched():
    df = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, np.nan]})
    out, _ = ist.impute_statistical(df, columns=["a"])
    assert out["a"].notna().all() and out["b"].isna().sum() == 1

def test_a_bad_setting_gives_a_clear_error():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError):
        ist.impute_statistical(df, config={"imputation": {"numeric_strategy": "magic"}})
    with pytest.raises(ValueError):
        ist.impute_statistical(df, config={"imputation": {"max_missing_shares": 0.5}})

def test_log_values_are_plain_json_types():
    import json
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "c": ["x", None, "x"]})
    _, log = ist.impute_statistical(df)
    json.dumps(log)

def test_mask_known_values_hides_the_requested_share():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    masked, truth, idx = ist.mask_known_values(df, "x", share=0.1, seed=1)
    assert len(idx) == 10 and masked["x"].isna().sum() == 10
    assert truth.notna().all()
    _, truth2, idx2 = ist.mask_known_values(df, "x", share=0.1, seed=1)
    assert list(idx) == list(idx2)          # same seed, same hidden values

def test_baseline_evaluation_scores_numeric_and_categorical_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "amount": rng.normal(100, 5, 200),
        "crop": ["Corn"] * 180 + ["Wheat"] * 20,
    })
    result = ist.evaluate_statistical(df, seed=0)
    assert result["amount"]["kind"] == "numeric" and result["amount"]["mae"] < 10
    assert result["crop"]["kind"] == "categorical" and result["crop"]["accuracy"] > 0.7

def test_baseline_evaluation_skips_tiny_columns():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert ist.evaluate_statistical(df)["x"]["kind"] == "skipped"