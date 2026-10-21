import pandas as pd
import pytest
import engine
import pii

CFG = {"severity": "high", "min_confidence": 0.5, "min_rows": 5, "sample_size": 5000,
       "regex_min_share": 0.5, "use_presidio": False,
       "weights": {"name": 0.55, "regex": 0.6, "presidio": 0.45, "semantic": 0.3},
       "name_hint_required": ["person_name"]}


def flagged(flags):
    return {f["column"]: f for f in flags}


def make_df():
    n = 30
    return pd.DataFrame({
        "email": [f"user{i}@example.com" for i in range(n)],
        "contact": [f"person{i}@mail.org" for i in range(n)],           # regex only
        "phone": [f"555-010-{1000 + i}" for i in range(n)],
        "full_name": [f"Person {i}" for i in range(n)],                 # name hint only
        "county_name": ["Polk", "Story", "Linn"] * 10,                  # not a person
        "payout_amount": [100.5 + i for i in range(n)],
        "loss_date": ["2024-01-05"] * n,
        "acres": [str(1000000 + i) for i in range(n)],                  # plain numbers
        "card": ["4539 1488 0343 6467"] * n,                            # passes Luhn
    })


def test_finds_the_expected_columns():
    got = flagged(pii.detect_pii(make_df(), CFG))
    assert set(got) == {"email", "contact", "phone", "full_name", "card"}


def test_ordinary_columns_are_not_flagged():
    got = flagged(pii.detect_pii(make_df(), CFG))
    for col in ["county_name", "payout_amount", "loss_date", "acres"]:
        assert col not in got


def test_two_signals_score_higher_than_one():
    got = flagged(pii.detect_pii(make_df(), CFG))
    assert got["email"]["confidence"] > got["contact"]["confidence"]
    assert got["contact"]["confidence"] > 0.5
    assert got["email"]["confidence"] <= 1


def test_a_single_signal_is_never_high_confidence():
    got = flagged(pii.detect_pii(make_df(), CFG))
    assert got["full_name"]["confidence"] < 0.7
    assert got["contact"]["confidence"] < 0.7


def test_flags_name_the_type_but_never_contain_values():
    text = str(pii.detect_pii(make_df(), CFG))
    for secret in ["user1@example.com", "555-010-1000", "Person 1", "4539"]:
        assert secret not in text
    assert "email" in text


def test_semantic_label_from_person_1_is_used_as_extra_signal():
    df = pd.DataFrame({"x": [f"a{i}@b.com" for i in range(10)]})
    meta = {"columns": {"x": {"semantic": "email"}}}
    without = flagged(pii.detect_pii(df, CFG))["x"]["confidence"]
    with_meta = flagged(pii.detect_pii(df, CFG, meta))["x"]["confidence"]
    assert with_meta > without


def test_picks_one_type_per_column():
    df = pd.DataFrame({"email_address": [f"u{i}@b.com" for i in range(10)]})
    flags = pii.detect_pii(df, CFG)
    assert len(flags) == 1 and "email" in flags[0]["evidence"]


def test_empty_dataframe_does_not_crash():
    result = pii.detect_pii(pd.DataFrame({"email": []}), CFG)
    # no crash; a column named 'email' may still be flagged by name
    assert isinstance(result, list)      


def test_missing_presidio_does_not_break_anything():
    # forget earlier attempts so the warning fires again
    pii._analyzer_cache.clear()    
    cfg = dict(CFG, use_presidio=True, presidio_model="model_that_does_not_exist")
    with pytest.warns(UserWarning, match="Presidio unavailable"):
        got = flagged(pii.detect_pii(make_df(), cfg))
    assert "email" in got


def test_works_through_the_engine_and_can_be_switched_off():
    on = {"version": 1, "rules": {"pii": dict(CFG)}}
    off = {"version": 1, "rules": {"pii": dict(CFG, enabled=False)}}
    assert engine.run_rules(make_df(), config=on)["total_flags"] > 0
    assert engine.run_rules(make_df(), config=off)["total_flags"] == 0