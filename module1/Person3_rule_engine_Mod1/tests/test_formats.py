import pandas as pd

import engine
import formats

CFG = {"severity": "medium", "confidence": 0.85, "min_rows": 20, "dominant_share": 0.95,
       "date_share": 0.8, "phone_share": 0.8, "max_shapes": 5,
       "max_distinct_ratio": 0.5, "sample_size": 50000}


def flagged(flags):
    return [f["column"] for f in flags]


def test_mixed_date_formats_are_flagged():
    dates = ["2024-01-05"] * 28 + ["05/01/2024"] * 2
    df = pd.DataFrame({"loss_date": dates, "other": ["x"] * 30})
    flags = formats.detect_format_inconsistency(df, CFG)
    assert flagged(flags) == ["loss_date"]
    assert "mixed date formats" in flags[0]["evidence"]


def test_consistent_dates_are_not_flagged():
    df = pd.DataFrame({"loss_date": [f"2024-01-{d:02d}" for d in range(1, 29)] * 2})
    assert formats.detect_format_inconsistency(df, CFG) == []


def test_mixed_casing_is_flagged():
    df = pd.DataFrame({"crop": ["Corn"] * 15 + ["CORN"] * 3 + ["Wheat"] * 12})
    flags = formats.detect_format_inconsistency(df, CFG)
    assert flagged(flags) == ["crop"]
    assert "casing" in flags[0]["evidence"]


def test_consistent_casing_is_not_flagged():
    df = pd.DataFrame({"crop": ["Corn", "Wheat", "Oats"] * 10})
    assert formats.detect_format_inconsistency(df, CFG) == []


def test_inconsistent_phone_patterns_are_flagged():
    phones = ["555-123-4567"] * 15 + ["(555) 123-4567"] * 10 + ["+1 555 123 4567"] * 5
    df = pd.DataFrame({"phone": phones})
    flags = formats.detect_format_inconsistency(df, CFG)
    assert flagged(flags) == ["phone"]
    assert "phone" in flags[0]["evidence"]


def test_consistent_phones_are_not_flagged():
    df = pd.DataFrame({"phone": ["555-123-4567", "555-987-6543"] * 15})
    assert formats.detect_format_inconsistency(df, CFG) == []


def test_lost_leading_zero_is_flagged_as_odd_shape():
    fips = ["01001", "19153", "17031", "20091"] * 12 + ["1001"]
    df = pd.DataFrame({"county_fips": fips})
    flags = formats.detect_format_inconsistency(df, CFG)
    assert flagged(flags) == ["county_fips"]


def test_free_text_and_names_are_not_flagged():
    df = pd.DataFrame({"note": [f"some different note number {i} here" * (1 + i % 3)
                                for i in range(40)]})
    assert formats.detect_format_inconsistency(df, CFG) == []


def test_numeric_columns_and_small_tables_are_skipped():
    big = pd.DataFrame({"n": range(50)})
    small = pd.DataFrame({"d": ["2024-01-05", "05/01/2024"]})
    assert formats.detect_format_inconsistency(big, CFG) == []
    assert formats.detect_format_inconsistency(small, CFG) == []


def test_flags_never_contain_real_values():
    df = pd.DataFrame({"crop": ["Zucchini"] * 15 + ["ZUCCHINI"] * 3 + ["Wheat"] * 12})
    flags = formats.detect_format_inconsistency(df, CFG)
    assert "Zucchini" not in str(flags) and "ZUCCHINI" not in str(flags)


def test_works_through_the_engine_and_can_be_switched_off():
    df = pd.DataFrame({"crop": ["Corn"] * 15 + ["CORN"] * 3 + ["Wheat"] * 12})
    on = {"version": 1, "rules": {"format_inconsistency": dict(CFG)}}
    off = {"version": 1, "rules": {"format_inconsistency": dict(CFG, enabled=False)}}
    assert engine.run_rules(df, config=on)["total_flags"] == 1
    assert engine.run_rules(df, config=off)["total_flags"] == 0

def test_phone_numbers_with_extensions_are_recognised():
    assert formats.looks_like_phone("(555)123-4567x123")
    assert formats.looks_like_phone("001-555-123-4567 ext. 89")
    assert not formats.looks_like_phone("2024-01-05")
    assert not formats.looks_like_phone("123-45-6789")