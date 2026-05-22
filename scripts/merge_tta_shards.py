#!/usr/bin/env python3
"""
Merge per-shard result CSVs from a sharded AlphaGenome TTA run into a single
results table + summary, recomputing the Pearson/Spearman correlations.

The sharded test scripts (``test_fulco_alphagenome.py`` / ``test_gasperini_alphagenome.py``
run with ``--num_shards N``) each write
``<output_prefix>_shard<KKK>of<NNN>_results.csv`` and skip the correlation step.
This script concatenates those shards and produces the same
``<output_prefix>_results.csv`` + ``<output_prefix>_summary.txt`` a single
(non-sharded) run would have written.

Usage:
    python scripts/merge_tta_shards.py \\
        --shards './results/test_Fulco_alphagenome_tta/Fulco_AlphaGenome_base_rna_seq_shard*of008_results.csv' \\
        --output_prefix Fulco_AlphaGenome_base_rna_seq \\
        --fulco_corr_observed_subset knockdown_only

For Gasperini, omit ``--fulco_corr_observed_subset`` (all pairs are used).
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def parse_args():
    p = argparse.ArgumentParser(description="Merge sharded AlphaGenome TTA result CSVs.")
    p.add_argument(
        "--shards", type=str, required=True,
        help="Glob (quote it!) matching the per-shard *_results.csv files.",
    )
    p.add_argument(
        "--output_prefix", type=str, required=True,
        help="Prefix for the merged <prefix>_results.csv / <prefix>_summary.txt.",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write the merged outputs (default: the dir of the first shard).",
    )
    p.add_argument(
        "--fulco_corr_observed_subset", type=str, default=None,
        choices=[None, "all", "knockdown_only"],
        help=(
            "Fulco only: restrict correlation rows. 'knockdown_only' keeps y_delta > 0. "
            "Omit (None) for Gasperini, which uses all pairs."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        raise FileNotFoundError("No shard CSVs matched glob: {}".format(args.shards))
    print("Merging {} shard files:".format(len(shard_paths)))
    for sp in shard_paths:
        print("  {}".format(sp))

    frames = [pd.read_csv(sp) for sp in shard_paths]
    df = pd.concat(frames, ignore_index=True)

    # Guard against accidental double-counting if a shard set overlaps.
    key_cols = [c for c in ("gene_id", "enh_loc", "tss") if c in df.columns]
    n_before = len(df)
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    if len(df) != n_before:
        print("Dropped {} duplicate rows across shards.".format(n_before - len(df)))
    print("Merged total: {} enhancer-gene pairs".format(len(df)))

    output_dir = args.output_dir or os.path.dirname(shard_paths[0]) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Correlation subset (Fulco knockdown_only vs all / Gasperini all)
    if args.fulco_corr_observed_subset == "knockdown_only":
        df_corr = df.loc[df["y_delta"].astype(float) > 0].copy()
    else:
        df_corr = df.copy()
    n_corr = len(df_corr)
    if n_corr < 2:
        raise ValueError("Need >= 2 pairs for correlation; got {}.".format(n_corr))

    pearson_r, pearson_p = pearsonr(df_corr["y_delta"], df_corr["pred_delta"])
    spearman_r, spearman_p = spearmanr(df_corr["y_delta"], df_corr["pred_delta"])

    have_log2 = "pred_wt" in df_corr.columns and "pred_crispri_mean" in df_corr.columns
    if have_log2:
        _eps = 1e-12
        expr_ratio = np.clip(1.0 - df_corr["y_delta"].astype(float).values, _eps, None)
        obs_log = -np.log2(expr_ratio)
        wt = np.maximum(df_corr["pred_wt"].astype(float).values, _eps)
        cr = np.maximum(df_corr["pred_crispri_mean"].astype(float).values, _eps)
        pred_log = np.log2(wt / cr)
        log_pear_r, log_pear_p = pearsonr(obs_log, pred_log)
        log_spr_r, log_spr_p = spearmanr(obs_log, pred_log)

    # Write merged results CSV
    csv_path = os.path.join(output_dir, "{}_results.csv".format(args.output_prefix))
    df.to_csv(csv_path, index=False)
    print("\nMerged results saved to: {}".format(csv_path))

    # Print + write summary
    print("\n" + "=" * 80)
    print("MERGED RESULTS SUMMARY")
    print("=" * 80)
    print("Pairs evaluated: {}   |   used for correlation: {}".format(len(df), n_corr))
    print("Pearson  (linear y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(pearson_r, pearson_p))
    print("Spearman (linear y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(spearman_r, spearman_p))
    if have_log2:
        print("Log2 alignment (-log2(expr_ratio) vs log2(pred_wt/pred_crispri)):")
        print("  Pearson  r: {:.4f}  (p={:.2e})".format(log_pear_r, log_pear_p))
        print("  Spearman r: {:.4f}  (p={:.2e})".format(log_spr_r, log_spr_p))
    print("=" * 80)

    summary_path = os.path.join(output_dir, "{}_summary.txt".format(args.output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MERGED TTA RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Shards merged:   {}\n".format(len(shard_paths)))
        f.write("Pairs evaluated: {}\n".format(len(df)))
        f.write("Pairs used for correlation: {}\n".format(n_corr))
        f.write("Observed subset: {}\n".format(args.fulco_corr_observed_subset or "all"))
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Pearson r  (linear):  {:.4f}  (p={:.2e})\n".format(pearson_r, pearson_p))
        f.write("Spearman r (linear):  {:.4f}  (p={:.2e})\n".format(spearman_r, spearman_p))
        if have_log2:
            f.write("Pearson r  (log2):    {:.4f}  (p={:.2e})\n".format(log_pear_r, log_pear_p))
            f.write("Spearman r (log2):    {:.4f}  (p={:.2e})\n".format(log_spr_r, log_spr_p))
        f.write("=" * 80 + "\n")
    print("Summary saved to: {}".format(summary_path))


if __name__ == "__main__":
    main()
