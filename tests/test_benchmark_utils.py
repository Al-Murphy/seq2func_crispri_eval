"""Unit tests for ``crispri_eval.benchmark_utils``."""

import pandas as pd
import pytest

from crispri_eval.benchmark_utils import fulco_results_for_correlation


@pytest.fixture
def small_results_df():
    """Three pairs: one knockdown, one no-effect, one increase."""
    return pd.DataFrame({
        "gene_name":   ["MYC", "GATA1", "HBE1"],
        "y_delta":     [0.45, 0.0, -0.10],
        "pred_delta":  [0.30, 0.05, -0.02],
    })


def test_all_keeps_every_row(small_results_df):
    out = fulco_results_for_correlation(small_results_df, "all")
    assert len(out) == 3
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        small_results_df.reset_index(drop=True),
    )


def test_knockdown_only_drops_zero_and_negative(small_results_df):
    out = fulco_results_for_correlation(small_results_df, "knockdown_only")
    assert len(out) == 1
    assert out["gene_name"].iloc[0] == "MYC"
    assert out["y_delta"].iloc[0] > 0


def test_returns_copy_not_view(small_results_df):
    """Mutating the result must NOT mutate the input — guards against pandas chained writes."""
    out = fulco_results_for_correlation(small_results_df, "all")
    out.loc[0, "y_delta"] = 999.0
    assert small_results_df.loc[0, "y_delta"] == 0.45


def test_invalid_subset_raises(small_results_df):
    with pytest.raises(ValueError, match="observed_subset"):
        fulco_results_for_correlation(small_results_df, "everything")


def test_string_y_delta_is_coerced():
    """``y_delta`` may arrive as a string after CSV round-trip; the helper must cope."""
    df = pd.DataFrame({"y_delta": ["0.5", "-0.1", "0.0"]})
    out = fulco_results_for_correlation(df, "knockdown_only")
    assert len(out) == 1
