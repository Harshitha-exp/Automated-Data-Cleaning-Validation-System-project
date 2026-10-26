"""
tests/test_metadata.py
------------------------
Pytest coverage for Person 1's metadata-extraction engine.

Run with: pytest tests/ -v
"""

import pandas as pd
import pytest

from person1_metadata.type_inference import infer_types
from person1_metadata.semantics import detect_semantics
from person1_metadata.mixed_types import detect_mixed_types
from person1_metadata.metadata import build_metadata


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "county_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "county_name": ["Autauga", "Autauga", "Baldwin", "Baldwin", "Barbour",
                             "Barbour", "Bibb", "Bibb", "Blount", "Blount"],
            "contact_email": [f"user{i}@example.com" for i in range(10)],
            "phone_number": ["555-123-4567", "555-234-5678", "555-345-6789",
                              "555-456-7890", "555-567-8901", "555-678-9012",
                              "555-789-0123", "555-890-1234", "555-901-2345", "555-012-3456"],
            "signup_date": ["2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20",
                             "2024-05-25", "2024-06-30", "2024-07-04", "2024-08-09",
                             "2024-09-14", "2024-10-19"],
            "payment_indemnity_usd": [1826.0, 49181.5, 9053.0, 12095.0, 27467.6,
                                       0.0, 44621.4, 1000.0, 15000.0, 22000.0],
            "zip_code": ["36003", "36003", "36502", "36502", "36420",
                         "36420", "35042", "35042", "35080", "35080"],
            "mixed_notes": ["1", "2", "abc", "4", "xyz", "6", "7", "not a number", "9", "10"],
            "all_null_col": [None] * 10,
        }
    )


def test_infer_types_structural_buckets(sample_df):
    types = infer_types(sample_df)
    assert types["county_id"]["type"] == "integer"
    assert types["payment_indemnity_usd"]["type"] == "float"
    assert types["all_null_col"]["type"] == "unknown"
    # low-cardinality object column -> categorical
    assert types["county_name"]["type"] == "categorical"


def test_detect_semantics_email(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["contact_email"]["semantic"] == "email"
    assert semantics["contact_email"]["confidence"] > 0.7


def test_detect_semantics_phone(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["phone_number"]["semantic"] == "phone"


def test_detect_semantics_date(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["signup_date"]["semantic"] == "date"


def test_detect_semantics_id(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["county_id"]["semantic"] == "id"


def test_detect_semantics_zip(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["zip_code"]["semantic"] == "zip"


def test_detect_semantics_currency(sample_df):
    semantics = detect_semantics(sample_df)
    assert semantics["payment_indemnity_usd"]["semantic"] == "currency"


def test_place_name_not_misread_as_person_name(sample_df):
    """'county_name' must NOT come back as person_name (regression test for
    the exact 'name' substring/token ambiguity both Person 1 and Person 3
    independently had to guard against)."""
    semantics = detect_semantics(sample_df)
    assert semantics["county_name"]["semantic"] != "person_name"


def test_detect_mixed_types_flags_mixed_column(sample_df):
    mixed = detect_mixed_types(sample_df)
    assert mixed["mixed_notes"]["is_mixed_type"] is True
    assert "integer" in mixed["mixed_notes"]["type_breakdown"]
    assert "string" in mixed["mixed_notes"]["type_breakdown"]


def test_detect_mixed_types_clean_column_not_flagged(sample_df):
    mixed = detect_mixed_types(sample_df)
    assert mixed["contact_email"]["is_mixed_type"] is False


def test_build_metadata_shape_matches_person3_contract(sample_df):
    """This is the load-bearing contract test: Person 3's column_rules.py
    reads metadata["columns"][col]["semantic"] — verify that exact path
    exists and is a plain lowercase string (or None), not a nested object
    or an enum/Mixed-case value."""
    meta = build_metadata(sample_df)
    assert "columns" in meta
    assert "county_id" in meta["columns"]
    semantic = meta["columns"]["county_id"]["semantic"]
    assert semantic == "id"
    assert isinstance(semantic, str)
    assert semantic == semantic.lower()


def test_build_metadata_covers_every_column(sample_df):
    meta = build_metadata(sample_df)
    assert set(meta["columns"].keys()) == set(sample_df.columns)
    assert meta["n_rows"] == 10
    assert meta["n_columns"] == 9


def test_all_null_column_has_no_confident_semantic(sample_df):
    meta = build_metadata(sample_df)
    assert meta["columns"]["all_null_col"]["semantic"] is None


# --- Regression tests for bugs found running against the real RMA dataset ---

def test_geographic_code_column_is_id_not_numeric():
    """'State'/'County' are low-cardinality FIPS codes (22 distinct values
    over a million rows) — they must be 'id', not a generic 'numeric'
    measure, even though cardinality_ratio alone (the usual id signal) is
    tiny for them, not huge."""
    df = pd.DataFrame({"State": [1, 1, 1, 17, 17, 17, 17, 48, 48, 48]})
    semantics = detect_semantics(df)
    assert semantics["State"]["semantic"] == "id"


def test_vague_money_word_does_not_trigger_currency_without_unit():
    """'Payment acreage' contains the word 'payment' but is a land-area
    measure, not money — must NOT be classified as currency. Only an
    explicit currency unit token (e.g. '(US$)' -> 'us') should."""
    df = pd.DataFrame({
        "Payment acreage": [79.2, 330.1, 107.0, 143.35, 539.66],
        "Payment indemnity (US$)": [1826.0, 49181.0, 9053.0, 12095.0, 27467.6],
    })
    semantics = detect_semantics(df)
    assert semantics["Payment acreage"]["semantic"] != "currency"
    assert semantics["Payment indemnity (US$)"]["semantic"] == "currency"
