"""
Full-pipeline integration tests: Person 1 -> Person 2 -> Person 3, exactly
as orchestrator.generate_report() wires them together, using everyone's
REAL code (not mocks). This is the load-bearing proof that the three
independently-built pieces actually fit together.
"""
import jsonschema
import pandas as pd
import pytest

from orchestrator import generate_report

import json
import os

# The shared contract lives at the repo root (schemas/), not inside the module,
# because other modules read the same file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHEMA_PATH = os.path.join(_REPO_ROOT, "schemas", "profiling_report.schema.json")


@pytest.fixture
def rma_like_df():
    """Shaped like the real RMA Cause-of-Loss dataset: geographic codes,
    a PII column, a constant column, and a skewed payout — one thing for
    every module to catch."""
    return pd.DataFrame(
        {
            # Fully unique (not constant) so it would normally trip
            # near_unique -- the point of the test below is that it
            # doesn't, because "State" is recognised as a geographic id.
            "State": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "County": [1, 1, 1, 3, 3, 5, 5, 7, 7, 9],
            "Year": [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
            "Cause of loss": ["Drought"] * 9 + ["Flood"],
            "Payment indemnity (US$)": [1826, 2000, 1950, 2100, 1800, 1900, 2050, 1975, 2200, 5_000_000],
            "contact_email": [f"farmer{i}@example.com" for i in range(10)],
            "always_same": ["X"] * 10,
        }
    )


def test_generate_report_runs_all_three_stages_with_zero_errors(rma_like_df):
    report = generate_report(rma_like_df)
    assert report["errors"] == []
    assert report["n_rows"] == 10
    assert report["n_columns"] == 7


def test_person1_metadata_feeds_person3_flags_correctly(rma_like_df):
    """State/County are geographic IDs (Person 1's job) and should
    therefore NOT trip Person 3's near_unique/skew rules; the genuinely
    constant and PII columns SHOULD be flagged."""
    report = generate_report(rma_like_df)
    assert report["metadata"]["columns"]["State"]["semantic"] == "id"

    flagged_columns = {f["column"] for f in report["flags"]}
    assert "always_same" in flagged_columns
    assert "contact_email" in flagged_columns
    assert "State" not in flagged_columns  # suppressed because semantic == "id"


def test_person2_statistics_and_visualisations_present(rma_like_df):
    report = generate_report(rma_like_df)
    assert report["statistics"]["missing"]["total_missing_cells"] == 0
    assert "Payment indemnity (US$)" in report["statistics"]["distributions"]
    assert set(report["visualisations"].keys()) == {
        "missing_heatmap", "outlier_distribution", "correlation_heatmap",
    }


def test_include_visualisations_false_skips_charts(rma_like_df):
    report = generate_report(rma_like_df, include_visualisations=False)
    assert report["visualisations"] == {}
    assert report["errors"] == []  # skipping charts is not an error


def test_report_validates_against_json_schema(rma_like_df):
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    report = generate_report(rma_like_df)
    jsonschema.validate(instance=report, schema=schema)
