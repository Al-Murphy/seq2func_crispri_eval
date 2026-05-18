"""
crispri_eval.benchmark_utils
============================

Row-selection helpers for correlating model predictions against measured
CRISPRi effects.
"""

import pandas as pd


def fulco_results_for_correlation(df: pd.DataFrame, observed_subset: str) -> pd.DataFrame:
    """
    Rows to use for Pearson/Spearman on Fulco benchmark result tables
    (``y_delta``, ``pred_delta``).

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark output with at least ``y_delta`` (``1 -`` Fulco expr ratio;
        > 0 = measured RNA decrease).
    observed_subset : {"all", "knockdown_only"}
        ``all``            — every row (same points as
                              ``plot_fulco_results.py --observed_mode signed``).
        ``knockdown_only`` — ``y_delta > 0`` only (same as
                              ``--observed_mode decrease_only``).

    Returns
    -------
    pd.DataFrame
    """
    if observed_subset not in ("all", "knockdown_only"):
        raise ValueError(
            "observed_subset must be 'all' or 'knockdown_only', got {!r}".format(
                observed_subset
            )
        )
    if observed_subset == "all":
        return df.copy()
    return df.loc[df["y_delta"].astype(float) > 0].copy()
