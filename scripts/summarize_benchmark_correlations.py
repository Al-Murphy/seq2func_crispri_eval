#!/usr/bin/env python3
"""
Summarise CRISPRi benchmark result CSVs into one correlation table.

For each ``label=path`` result CSV it reports the same Pearson/Spearman that the
test scripts print, on ``y_delta`` vs ``pred_delta``:

  * ``--benchmark fulco``     -> knockdown-only rows (``y_delta > 0``), matching
                                 ``--fulco_corr_observed_subset knockdown_only`` /
                                 ``plot_fulco_results.py --observed_mode decrease_only``.
  * ``--benchmark gasperini`` -> all rows (Gasperini result CSVs are already the
                                 high-confidence subset).

When a CSV carries ``pred_wt`` / ``pred_crispri_mean`` it also reports the log2
knockdown-strength correlation (−log2(1−y_delta) vs log2(pred_wt/pred_crispri)),
the secondary metric the test scripts emit.

This is meant for assembling the per-model / per-fold / ensemble comparison table
(e.g. Borzoi folds 0-3 vs the 4-fold ensemble) as a blog supplementary file.

Usage:
    python scripts/summarize_benchmark_correlations.py \\
        --benchmark fulco \\
        --results \\
          "Enformer=./results/test_Fulco_enformer/Fulco_base_base_enformer_results.csv" \\
          "Borzoi fold0=./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv" \\
          "Borzoi ensemble=./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv" \\
        --output_csv ./results/test_Fulco_borzoi/fulco_correlation_summary.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_EPS_LOG2 = 1e-12


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarise benchmark result CSVs into one correlation table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--benchmark", choices=["fulco", "gasperini"], required=True,
        help="Selects the row subset used for correlation (see module docstring).",
    )
    p.add_argument(
        "--results", type=str, nargs="+", required=True,
        help="One or more 'Label=path/to/results.csv' entries (label may contain spaces).",
    )
    p.add_argument(
        "--output_csv", type=str, default=None,
        help="Optional path to write the summary table as CSV.",
    )
    return p.parse_args()


def _split_label_path(entry):
    if "=" not in entry:
        raise ValueError("Expected 'Label=path', got {!r}.".format(entry))
    label, path = entry.split("=", 1)
    return label.strip(), path.strip()


def _subset(df, benchmark):
    if benchmark == "fulco":
        return df.loc[df["y_delta"].astype(float) > 0].copy()
    return df.copy()


def summarize_one(label, path, benchmark):
    df = pd.read_csv(path)
    df = _subset(df, benchmark)
    n = len(df)
    row = {"label": label, "n": n, "path": path}
    if n < 2:
        return row

    y = df["y_delta"].astype(float).values
    pred = df["pred_delta"].astype(float).values
    row["pearson_r"], row["pearson_p"] = pearsonr(y, pred)
    row["spearman_r"], row["spearman_p"] = spearmanr(y, pred)

    if "pred_wt" in df.columns and "pred_crispri_mean" in df.columns:
        expr_ratio = np.clip(1.0 - y, _EPS_LOG2, None)
        obs_log2 = -np.log2(expr_ratio)
        wt = np.maximum(df["pred_wt"].astype(float).values, _EPS_LOG2)
        cr = np.maximum(df["pred_crispri_mean"].astype(float).values, _EPS_LOG2)
        pred_log2 = np.log2(wt / cr)
        row["log2_pearson_r"], row["log2_pearson_p"] = pearsonr(obs_log2, pred_log2)
        row["log2_spearman_r"], row["log2_spearman_p"] = spearmanr(obs_log2, pred_log2)
    return row


def _fmt(v):
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0:
            return "{:.2e}".format(v)
        return "{:.4f}".format(v)
    return "" if v is None else str(v)


def print_markdown(rows):
    cols = ["label", "n", "pearson_r", "spearman_r"]
    if any("log2_pearson_r" in r for r in rows):
        cols += ["log2_pearson_r", "log2_spearman_r"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")


def main():
    args = parse_args()
    rows = []
    for entry in args.results:
        label, path = _split_label_path(entry)
        if not os.path.isfile(path):
            raise FileNotFoundError("Result CSV not found for '{}': {}".format(label, path))
        rows.append(summarize_one(label, path, args.benchmark))

    print("\nBenchmark: {}  (subset: {})\n".format(
        args.benchmark, "knockdown_only (y_delta>0)" if args.benchmark == "fulco" else "all rows"))
    print_markdown(rows)

    if args.output_csv:
        out = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        out.to_csv(args.output_csv, index=False)
        print("\nSummary CSV written to: {}".format(args.output_csv))


if __name__ == "__main__":
    main()
