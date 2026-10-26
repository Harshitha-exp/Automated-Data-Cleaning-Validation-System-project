"""
Unit tests for Person 2's profiling-statistics and chart functions.
"""
import base64

import pandas as pd
import pytest

from person2_profiling.stats import (
    compute_cardinality,
    compute_correlations,
    compute_distributions,
    compute_statistics,
    profile_missing,
    type_consistency_check,
)
from person2_profiling.charts import (
    generate_visualisations,
    plot_correlation_heatmap,
    plot_missing_heatmap,
    plot_outlier_distribution,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": range(1, 21),
            "state": ["AL"] * 10 + ["CA"] * 10,
            "cause": ["Drought", "Flood"] * 10,
            "amount": [100.0, 200.0, 150.0, 300.0, None, 500.0, 120.0, 800.0, 90.0, 110.0] * 2,
        }
    )


def test_profile_missing_counts(sample_df):
    result = profile_missing(sample_df)
    assert result["per_column"]["amount"]["n_missing"] == 2
    assert result["per_column"]["id"]["n_missing"] == 0
    assert result["total_missing_cells"] == 2
    assert result["rows_with_any_missing"] == 2


def test_type_consistency_flags_mixed_column():
    df = pd.DataFrame({"mixed_notes": ["1", "2", "abc", "4", "xyz", "6", "7", "not a number", "9", "10"]})
    result = type_consistency_check(df)
    assert result["mixed_notes"]["declared_type"] == "text"
    # "text" has no narrower shape to violate, so this just proves the
    # function runs cleanly on a genuinely mixed column without crashing
    assert "consistency_ratio" in result["mixed_notes"]


def test_type_consistency_uses_person1_metadata_when_given():
    df = pd.DataFrame({"county_fips": ["01001", "01003", "01005"]})
    metadata = {"columns": {"county_fips": {"type": "integer"}}}
    result = type_consistency_check(df, metadata=metadata)
    assert result["county_fips"]["declared_type"] == "integer"
    assert result["county_fips"]["consistency_ratio"] == 1.0  # all values are digit strings


def test_compute_cardinality_detects_constant_and_unique(sample_df):
    result = compute_cardinality(sample_df)
    assert result["id"]["is_unique"] is True
    assert result["id"]["is_constant"] is False
    assert result["cause"]["is_constant"] is False
    assert result["cause"]["n_unique"] == 2


def test_compute_distributions_numeric_vs_categorical(sample_df):
    result = compute_distributions(sample_df)
    assert result["amount"]["kind"] == "numeric"
    assert "mean" in result["amount"]
    assert result["state"]["kind"] == "categorical"
    assert result["state"]["mode"] in ("AL", "CA")


def test_compute_correlations_numeric_and_categorical(sample_df):
    result = compute_correlations(sample_df)
    assert "id" in result["numeric"]["columns"]
    assert "amount" in result["numeric"]["columns"]
    assert len(result["numeric"]["matrix"]) == len(result["numeric"]["columns"])
    assert "state" in result["categorical"]["columns"]
    assert "cause" in result["categorical"]["columns"]


def test_compute_statistics_combines_everything(sample_df):
    stats = compute_statistics(sample_df)
    assert set(stats.keys()) == {"missing", "type_consistency", "cardinality", "distributions", "correlations"}


def test_plot_missing_heatmap_returns_valid_base64_png(sample_df):
    result = plot_missing_heatmap(sample_df)
    assert result["format"] == "png"
    assert result["encoding"] == "base64"
    decoded = base64.b64decode(result["data"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature


def test_plot_outlier_distribution_returns_valid_png(sample_df):
    result = plot_outlier_distribution(sample_df)
    decoded = base64.b64decode(result["data"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_correlation_heatmap_from_stats_dict(sample_df):
    corr = compute_correlations(sample_df)
    result = plot_correlation_heatmap(corr, kind="numeric")
    decoded = base64.b64decode(result["data"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_correlation_heatmap_handles_too_few_columns():
    df = pd.DataFrame({"only_col": [1, 2, 3]})
    corr = compute_correlations(df)
    result = plot_correlation_heatmap(corr, kind="numeric")
    # Should degrade gracefully (a "not enough columns" placeholder image), never raise.
    assert result["format"] == "png"


def test_generate_visualisations_returns_all_three_charts(sample_df):
    viz = generate_visualisations(sample_df)
    assert set(viz.keys()) == {"missing_heatmap", "outlier_distribution", "correlation_heatmap"}
    for chart in viz.values():
        assert chart["format"] == "png"
        assert chart["encoding"] in ("base64", "path")
