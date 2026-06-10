#!/usr/bin/env python3
"""
Build a Borzoi/Flashzoi *ensemble* result CSV from the per-fold (replicate)
result CSVs produced by ``test_fulco_borzoi.py`` / ``test_gasperini_borzoi.py``.

Borzoi/Flashzoi ship four trained replicates (folds 0-3). The published way to
"ensemble" them is to average the model outputs across replicates and then derive
downstream quantities -- NOT to average the ratio ``pred_delta`` directly. So this
script:

  * matches pairs across the per-fold CSVs on (gene_id, enh_loc, tss),
  * averages ``pred_wt`` and ``pred_crispri_mean`` across folds, and
  * recomputes ``pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)``
    exactly as the test scripts do.

Identifier / label columns (gene_name, enh_dist, y_delta, strand, ...) are taken
from the first input and asserted identical across folds. The output is a drop-in
result CSV with the same schema, so ``plot_fulco_results.py`` /
``plot_gasperini_results.py`` (incl. their log2 panels, which read pred_wt /
pred_crispri_mean) consume it unchanged.

Usage:
    python scripts/make_borzoi_ensemble.py \\
        --inputs ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \\
                 ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_rep1_results.csv \\
                 ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_rep2_results.csv \\
                 ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_rep3_results.csv \\
        --output ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

PRED_DELTA_EPS = 1e-8

KEY_COLS = ["gene_id", "enh_loc", "tss"]
# Averaged across folds (the actual model outputs).
MEAN_COLS = ["pred_wt", "pred_crispri_mean"]
# Must be identical across folds (same dataset / pairs); carried from fold 0.
IDENTITY_COLS = [
    "gene_name", "tss_seq_index", "tss_bin", "strand", "enh_dist", "y_delta",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Average per-fold Borzoi/Flashzoi result CSVs into an ensemble CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--inputs", type=str, nargs="+", required=True,
        help="Per-fold result CSVs (e.g. the rep0/rep1/rep2/rep3 flashzoi files).",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Path for the ensemble result CSV.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Need at least two per-fold CSVs to ensemble; got {}.".format(len(args.inputs)))

    missing = [f for f in args.inputs if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError("Input CSV(s) not found:\n  " + "\n  ".join(missing))

    dfs = []
    for f in args.inputs:
        df = pd.read_csv(f)
        for col in KEY_COLS + MEAN_COLS:
            if col not in df.columns:
                raise KeyError("Column '{}' missing in {}.".format(col, f))
        # Index by the pair key so folds align even if row order differs.
        df = df.set_index(KEY_COLS).sort_index()
        dfs.append((f, df))

    base_path, base = dfs[0]
    n_folds = len(dfs)

    # All folds must cover exactly the same pairs.
    for f, df in dfs[1:]:
        if not base.index.equals(df.index):
            only_base = base.index.difference(df.index)
            only_other = df.index.difference(base.index)
            raise ValueError(
                "Pair keys differ between\n  {}\nand\n  {}\n"
                "  {} only in first, {} only in second (e.g. {}).".format(
                    base_path, f, len(only_base), len(only_other),
                    list(only_base[:3]) + list(only_other[:3]),
                )
            )

    out = base.reset_index().copy()

    # Identity columns: assert agreement across folds, then keep fold 0's values.
    for col in IDENTITY_COLS:
        if col not in base.columns:
            continue
        for f, df in dfs[1:]:
            if col not in df.columns:
                continue
            a = base[col].to_numpy()
            b = df.loc[base.index, col].to_numpy()
            if a.dtype.kind in "fc" or b.dtype.kind in "fc":
                same = np.allclose(np.asarray(a, float), np.asarray(b, float),
                                   rtol=0, atol=1e-6, equal_nan=True)
            else:
                same = bool(np.array_equal(a, b))
            if not same:
                raise ValueError(
                    "Identity column '{}' differs between {} and {}; folds are not "
                    "aligned to the same pairs/labels.".format(col, base_path, f)
                )

    # Average the model outputs across folds, then recompute pred_delta.
    stacked = {col: np.vstack([df.loc[base.index, col].to_numpy(float) for _, df in dfs])
               for col in MEAN_COLS}
    for col in MEAN_COLS:
        out[col] = stacked[col].mean(axis=0)

    out["pred_delta"] = (
        (out["pred_wt"] - out["pred_crispri_mean"]) / (out["pred_wt"] + PRED_DELTA_EPS)
    )

    # Preserve fold 0's column order; drop any per-fold debug spread columns that
    # are no longer meaningful after averaging.
    drop_cols = [c for c in base.reset_index().columns
                 if c.startswith("debug_crispri_")]
    col_order = [c for c in base.reset_index().columns if c not in drop_cols]
    out = out[col_order]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Ensembled {} folds over {} pairs:".format(n_folds, len(out)))
    for f, _ in dfs:
        print("  - {}".format(f))
    print("Wrote ensemble result CSV: {}".format(args.output))


if __name__ == "__main__":
    main()
