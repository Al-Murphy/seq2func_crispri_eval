#!/usr/bin/env python3
"""
Plot Gasperini benchmark results comparing CRISPRi effects with model predictions.

This script creates visualizations of enhancer-gene distance vs change in expression,
comparing observed CRISPRi effects with model predictions. Can handle single or 
multiple model results as subplots.

Usage:
    # Single model
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
        --output ./results/test_Gasperini_enformer/plots/gasperini_base.png
    
    # Multiple models - 49k enf and lucidrains enformer (creates subplots)
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_enformer/Gasperini_finetuned_base_CL_enf_49k_human_head_results.csv \
        --labels "Enformer" "Enformer 49k" \
        --output ./results/test_Gasperini_enformer/plots/gasperini_comparison_49k_enf.png
        
    # Multiple models - 49k enf, lucidrains enformer and decima (creates subplots)
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_enformer/Gasperini_finetuned_base_CL_enf_49k_human_head_results.csv \
                  ./results/test_Gasperini_enformer/Gasperini_Decima_results.csv \
        --labels "Enformer" "Enformer 49k" "Decima Ensemble" \
        --output ./results/test_Gasperini_enformer/plots/gasperini_comparison_49k_enf_decima.png

    # AlphaGenome base model 
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_cage_results.csv \
        --labels "Enformer" "AlphaGenome" \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_ag_base.png
    
    # AlphaGenome CAGE/RNA base model 
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_cage_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "AlphaGenome CAGE" "AlphaGenome RNA-Seq" \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_ag_base_cage_rna.png

    #100 kb
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_cage_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "AlphaGenome CAGE" "AlphaGenome RNA-Seq"\
        --max_enh_dist 98304 \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_ag_base_cage_rna_100kb.png
        
    #100 kb
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "Borzoi" "AlphaGenome"\
        --max_enh_dist 98304 \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_borzoi_ag_100kb.png   
    
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "Borzoi" "AlphaGenome"\
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_borzoi_ag.png     
        
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv \
                  ./results/test_Gasperini_ntv3/Gasperini_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "Borzoi" "NTv3" "AlphaGenome" \
        --max_width_subplots 4 \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_borzoi_ntv3_ag.png 
    
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv \
                  ./results/test_Gasperini_ntv3/Gasperini_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
        --labels "Enformer" "Borzoi" "NTv3" "AlphaGenome" \
        --max_width_subplots 4 \
        --max_enh_dist 98304 \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_borzoi_ntv3_ag_100kb.png                 

    # Same as above but only pairs within 100 kb of TSS (fair Enformer vs AlphaGenome window)
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_cage_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_tss_results.csv \
        --labels "Enformer" "AlphaGenome CAGE" "AlphaGenome RNA-Seq" "AlphaGenome RNA-Seq (TSS window)"\
        --max_enh_dist 98304 \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_enf_ag_base_cage_rna_tss_100kb.png

    # AlphaGenome: base vs fine-tuned (original head) vs fine-tuned (CRISPRi head)
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_finetuned_CTLA4_..._head_original_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_finetuned_CTLA4_..._head_crispri_results.csv \
        --labels "AlphaGenome Base" "AlphaGenome FT (original head)" "AlphaGenome FT (CRISPRi head)" \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_ag_comparison.png

    # All models side-by-side
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_enformer/Gasperini_base_base_enformer_results.csv \
                  ./results/test_Gasperini_enformer/Gasperini_Decima_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_finetuned_CTLA4_..._head_original_results.csv \
        --labels "Enformer" "Decima" "AlphaGenome Base" "AlphaGenome FT (original head)" \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_all_models.png

    # Log2 scatter (auto-generated as _log2_scatter.png for any CSV with pred_wt/pred_crispri_mean).
    # To compare RNA-Seq exon-mean vs TSS-window vs CAGE, run test_gasperini_alphagenome.py
    # with three configurations and pass all three CSVs here:
    #   1. Default (exon-mean): produces Gasperini_AlphaGenome_base_rna_seq_results.csv
    #   2. --exon_csv none (TSS 640 bp window): --output_prefix Gasperini_AlphaGenome_base_rna_tss
    #   3. --modality cage: produces Gasperini_AlphaGenome_base_cage_results.csv
    python scripts/3_benchmark/plot_gasperini_results.py \
        --results ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_tss_results.csv \
                  ./results/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_cage_results.csv \
        --labels "RNA-Seq (all exons)" "RNA-Seq (TSS 640 bp)" "CAGE (TSS 640 bp)" \
        --output ./results/test_Gasperini_alphagenome/plots/gasperini_tss_vs_exon.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def derive_scatter_output_path(output_path):
    """Create default companion output path for effect scatter panels."""
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".png"
    return "{}_effect_scatter{}".format(root, ext)


def setup_plot_style():
    """Set up the plotting style and color palette."""
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")
    
    # Color palette
    pal = ["#8B9DAF", "#A65141", "#1d6cb1", "#394165", "#80A0C7", "#E7CDC2",
           "#B1934A", "#DCA258", "#EEDA9D", "#E8DCCF"]
    
    return pal


# Log y-axis: truncated predictions clipped to 0 are plotted at this y (log cannot show 0).
LOG_TRUNCATED_PRED_Y = 0.001


def _y_for_log_plot(y):
    """
    Matplotlib log scale requires strictly positive y. Map non-positive values to a
    small floor so truncated zeros (and any non-positive observations) remain visible.

    Floor is min(positive y) * 1e-4 when positives exist, else 1e-8 (display only).
    Truncated model points clipped to 0 use ``LOG_TRUNCATED_PRED_Y`` on log axes (see plot loop).
    """
    y = np.asarray(y, dtype=float)
    out = y.copy()
    nonpos = out <= 0
    if not np.any(nonpos):
        return out
    pos = out[~nonpos]
    eps = max(float(np.min(pos)) * 1e-4, 1e-15) if pos.size > 0 else 1e-8
    out[nonpos] = eps
    return out


def filter_enh_dist(df, max_bp, min_bp, source_label=None):
    """
    Keep rows with ``enh_dist`` within [min_bp, max_bp].

    Parameters
    ----------
    max_bp : int or None – upper bound (inclusive). None = no upper limit.
    min_bp : int or None – lower bound (exclusive). None = no lower limit.
        SequenceModelBenchmark uses > 990 bp to exclude near-promoter elements.
    source_label : str or None
        Printed in the log line.
    """
    if max_bp is None and min_bp is None:
        return df
    if "enh_dist" not in df.columns:
        raise KeyError(
            "CSV must contain an 'enh_dist' column to use --max_enh_dist / --min_enh_dist "
            "(Gasperini benchmark result files include it)."
        )
    n0 = len(df)
    dist = df["enh_dist"].astype(float)
    mask = pd.Series(True, index=df.index)
    if max_bp is not None:
        mask &= dist <= float(max_bp)
    if min_bp is not None:
        mask &= dist > float(min_bp)
    out = df[mask].copy()
    n1 = len(out)
    tag = " ({})".format(source_label) if source_label else ""
    parts = []
    if min_bp is not None:
        parts.append("min_enh_dist>{:,}".format(int(min_bp)))
    if max_bp is not None:
        parts.append("max_enh_dist<={:,}".format(int(max_bp)))
    print("dist filter [{}] bp{}: kept {}/{} pairs".format(", ".join(parts), tag, n1, n0))
    if n1 == 0:
        raise ValueError(
            "No rows left after enh_dist filter; adjust --min_enh_dist / --max_enh_dist."
        )
    return out


def process_dataframe(df):
    """
    ``pred_delta_raw`` — values from the CSV (used only to fit the regression line on in-range points).
    ``pred_delta`` — clipped to [0, 1] for **scatter** display.
    ``truncated`` — True if raw value was outside [0, 1] (excluded from the fitted line).
    """
    df = df.copy()
    df["pred_delta_raw"] = df["pred_delta"].astype(float)
    df["truncated"] = (df["pred_delta_raw"] < 0) | (df["pred_delta_raw"] > 1)
    df["pred_delta"] = df["pred_delta_raw"].apply(
        lambda x: 0.0 if x < 0 else (1.0 if x > 1 else x)
    )
    return df


def plot_single_model(df, ax, pal, title=None, show_legend=True, show_ylabel=True, y_scale='linear'):
    """Plot a single model's results on given axes."""
    log_y = y_scale == "log"

    # Observed: for log y, map non-positive values to a display floor (log cannot plot 0)
    if log_y:
        df_obs = df.assign(_y_obs=_y_for_log_plot(df["y_delta"].values))
        sns.regplot(
            data=df_obs,
            x="enh_dist",
            y="_y_obs",
            scatter_kws={"alpha": 1, "s": 10},
            line_kws={"linewidth": 3},
            color=pal[0],
            label="Observed CRISPRi Effect",
            ax=ax,
        )
    else:
        sns.regplot(
            data=df,
            x="enh_dist",
            y="y_delta",
            scatter_kws={"alpha": 1, "s": 10},
            line_kws={"linewidth": 3},
            color=pal[0],
            label="Observed CRISPRi Effect",
            ax=ax,
        )

    # Regression line: only non-truncated points; for log y, floor non-positive fit y
    df_fit = df[~df["truncated"]]
    if len(df_fit) >= 2:
        if log_y:
            df_line = df_fit.assign(
                _y_fit=_y_for_log_plot(df_fit["pred_delta_raw"].values)
            )
            sns.regplot(
                data=df_line,
                x="enh_dist",
                y="_y_fit",
                scatter=False,
                line_kws={"linewidth": 3},
                color=pal[2],
                label="_nolegend_",
                ax=ax,
            )
        else:
            sns.regplot(
                data=df_fit,
                x="enh_dist",
                y="pred_delta_raw",
                scatter=False,
                line_kws={"linewidth": 3},
                color=pal[2],
                label="_nolegend_",
                ax=ax,
            )
    elif len(df_fit) == 1:
        y0 = float(df_fit["pred_delta_raw"].iloc[0])
        if log_y:
            y0 = float(_y_for_log_plot(np.array([y0]))[0])
        ax.axhline(y0, color=pal[2], linewidth=3)

    # Scatter uses truncated pred_delta; floor zeros for log y so clipped points are visible
    df_ok = df[~df["truncated"]]
    df_clip = df[df["truncated"]]
    if len(df_ok) > 0:
        y_ok = _y_for_log_plot(df_ok["pred_delta"].values) if log_y else df_ok["pred_delta"]
        ax.scatter(
            df_ok["enh_dist"],
            y_ok,
            alpha=1,
            s=10,
            marker="o",
            color=pal[2],
            zorder=5,
            label="Model Predictions",
        )
    if len(df_clip) > 0:
        if log_y:
            pdv = df_clip["pred_delta"].values.astype(float)
            y_clip = np.where(pdv <= 0, LOG_TRUNCATED_PRED_Y, pdv)
        else:
            y_clip = df_clip["pred_delta"]
        ax.scatter(
            df_clip["enh_dist"],
            y_clip,
            alpha=1,
            s=10,
            marker="^",
            color=pal[2],
            zorder=5,
            label="Model Predictions (Truncated)",
        )

    ax.set_yscale(y_scale)
    
    # For log scale, format tick labels as natural numbers instead of scientific notation
    if y_scale == 'log':
        # Format y-axis labels as natural numbers (100, 10, 1, 0.1) instead of scientific notation
        def format_log_label(x, p):
            """Format log scale tick labels as natural numbers."""
            if x >= 1:
                return f'{x:.0f}'
            elif x >= 0.1:
                return f'{x:.1f}'
            else:
                # For values < 0.1, show one decimal place
                return f'{x:.2f}'
        
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_log_label))
    
    ax.set_xlabel('Distance from enhancer to TSS (bp)', fontsize=12)
    
    if show_ylabel:
        ax.set_ylabel('Change in Expression (%)\ndue to Enhancer Knockdown', fontsize=12)
    else:
        ax.set_ylabel('', fontsize=0)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('CRISPRi Effect vs Model Predictions by Distance from TSS', fontsize=14)
    
    if show_legend:
        ax.legend(fontsize=11, bbox_to_anchor=(1, 0.8), loc='upper left', frameon=False)
    ax.grid(True, alpha=0.3)


_EPS_LOG2 = 1e-12


def derive_log2_output_path(output_path):
    """Auto-generate companion path for the log2 scatter figure."""
    root, ext = os.path.splitext(output_path)
    return "{}_log2_scatter{}".format(root, ext or ".png")


def derive_log2_dist_output_path(output_path):
    """Auto-generate companion path for the log2 distance figure."""
    root, ext = os.path.splitext(output_path)
    return "{}_log2_dist{}".format(root, ext or ".png")


def _has_log2_cols(df):
    """True when pred_wt and pred_crispri_mean are present (new-format results)."""
    return "pred_wt" in df.columns and "pred_crispri_mean" in df.columns


def compute_log2_effects(df):
    """Return a copy of *df* with two new columns:

    ``obs_log2``   −log₂(1 − y_delta)  — Karollus knockdown strength
                   (positive = expression fell; larger = stronger effect).
    ``pred_log2``  log₂(pred_wt / pred_crispri_mean)
                   (positive = model predicts knockdown).
    """
    df = df.copy()
    expr_ratio = np.clip(1.0 - df["y_delta"].astype(float).values, _EPS_LOG2, None)
    df["obs_log2"] = -np.log2(expr_ratio)
    wt = np.maximum(df["pred_wt"].astype(float).values, _EPS_LOG2)
    cr = np.maximum(df["pred_crispri_mean"].astype(float).values, _EPS_LOG2)
    df["pred_log2"] = np.log2(wt / cr)
    return df


def _plot_log2_dist_single(df, ax, pal, title=None, show_legend=True, show_ylabel=True):
    """Plot enh_dist vs log₂ effect for one model on *ax*.

    Blue points/line  — observed: −log₂(1 − y_delta)
    Orange points/line — predicted: log₂(pred_wt / pred_crispri_mean)
    """
    sns.regplot(
        data=df, x="enh_dist", y="obs_log2",
        scatter_kws={"alpha": 0.8, "s": 10},
        line_kws={"linewidth": 2.5},
        color=pal[0], label="Observed log₂ knockdown", ax=ax,
    )
    sns.regplot(
        data=df, x="enh_dist", y="pred_log2",
        scatter_kws={"alpha": 0.8, "s": 10},
        line_kws={"linewidth": 2.5},
        color=pal[2], label="Predicted log₂ knockdown", ax=ax,
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Distance from enhancer to TSS (bp)", fontsize=12)
    if show_ylabel:
        ax.set_ylabel(
            "log₂ knockdown strength\n"
            "(obs: −log₂(1−y_delta);  pred: log₂(WT/CRISPRi))",
            fontsize=11,
        )
    else:
        ax.set_ylabel("", fontsize=0)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("log₂ CRISPRi Effect vs Distance from TSS", fontsize=14)
    if show_legend:
        ax.legend(fontsize=11, bbox_to_anchor=(1, 0.8), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)


def plot_log2_dist_panels(
    result_files,
    labels,
    output_path,
    dpi,
    max_enh_dist=None,
    min_enh_dist=None,
    max_width_subplots=3,
):
    """Distance-style plot (enh_dist × log₂ effect) for each model.

    Mirrors the original distance plot but uses log₂(pred_WT/pred_CRISPRi) and
    −log₂(1−y_delta) on the y-axis so that the near-zero linear effects are
    resolved into a more informative scale.

    CSVs without ``pred_wt``/``pred_crispri_mean`` are skipped with a warning.
    """
    pal = setup_plot_style()

    valid = []
    for path, label in zip(result_files, labels):
        df = pd.read_csv(path)
        if not _has_log2_cols(df):
            print(
                "  [log2 dist] skipping '{}' — no pred_wt/pred_crispri_mean columns".format(
                    label or path)
            )
            continue
        valid.append((path, label, df))

    if not valid:
        print("  [log2 dist] no eligible CSVs; skipping log2 distance plot")
        return

    n_models = len(valid)
    if n_models <= max_width_subplots:
        nrows, ncols = 1, n_models
        figsize = (10 * n_models, 6)
    else:
        ncols = max_width_subplots
        nrows = (n_models + ncols - 1) // ncols
        figsize = (30, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    ylabel_shared = (
        "log₂ knockdown strength\n"
        "(obs: −log₂(1−y_delta);  pred: log₂(WT/CRISPRi))"
    )

    for plot_idx, (path, label, df_raw) in enumerate(valid):
        ax = axes_flat[plot_idx]
        df = filter_enh_dist(df_raw, max_enh_dist, min_enh_dist, source_label=label or path)
        df = compute_log2_effects(df)
        _plot_log2_dist_single(df, ax, pal, title=label,
                               show_legend=False, show_ylabel=False)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.text(0.04, 0.5, ylabel_shared, va="center", rotation="vertical", fontsize=11)

    handles, legend_labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=11,
               bbox_to_anchor=(0.85, 0.5), loc="center left", frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.85)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print("Log2 distance plot saved to: {}".format(output_path))
    plt.close()


def plot_log2_scatter_panels(
    result_files,
    labels,
    output_path,
    dpi,
    max_enh_dist=None,
    min_enh_dist=None,
    max_width_subplots=3,
):
    """Scatter of −log₂(1−y_delta) (x) vs log₂(pred_WT/pred_CRISPRi) (y), coloured by enh_dist.

    Only CSVs that contain ``pred_wt`` and ``pred_crispri_mean`` columns are
    included.  CSVs from older runs (linear pred_delta only) are silently
    skipped with a warning.

    Tip — comparing TSS-window vs all-exon RNA-Seq scoring:
      Run test_gasperini_alphagenome.py twice:
        1. Default  → exon-mean aggregation (pred_wt reflects all exons)
        2. ``--exon_csv none`` → 640 bp TSS-window (same window as CAGE)
      Pass both result CSVs (plus the CAGE CSV) with different labels; this
      function shows all panels side-by-side.
    """
    from scipy.stats import pearsonr, spearmanr
    from matplotlib.colors import LogNorm, Normalize

    pal = setup_plot_style()

    panel_data = []
    all_dist = []
    all_y = []
    for path, label in zip(result_files, labels):
        df = pd.read_csv(path)
        if not _has_log2_cols(df):
            print(
                "  [log2 scatter] skipping '{}' — no pred_wt/pred_crispri_mean columns "
                "(re-run test script to get them)".format(label or path)
            )
            continue
        df = filter_enh_dist(df, max_enh_dist, min_enh_dist, source_label=label or path)
        df = compute_log2_effects(df)
        panel_data.append((path, label, df))
        all_dist.append(df["enh_dist"].astype(float).values)
        all_y.append(df["pred_log2"].values)

    if not panel_data:
        print("  [log2 scatter] no eligible CSVs; skipping log2 scatter output")
        return

    n_models = len(panel_data)
    ncols = min(n_models, max_width_subplots)
    nrows = (n_models + ncols - 1) // ncols
    figsize = (6 * ncols, 5.5 * nrows)

    dist_concat = np.concatenate(all_dist) if all_dist else np.array([1.0])
    finite_dist = dist_concat[np.isfinite(dist_concat) & (dist_concat > 0)]
    if finite_dist.size >= 2 and finite_dist.min() < finite_dist.max():
        norm = LogNorm(vmin=max(float(finite_dist.min()), 1.0),
                       vmax=float(finite_dist.max()))
    else:
        vmax = float(finite_dist.max()) if finite_dist.size else 1.0
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = plt.cm.viridis

    y_concat = np.concatenate(all_y) if all_y else np.array([0.0])
    finite_y = y_concat[np.isfinite(y_concat)]
    if finite_y.size:
        y_lo, y_hi = float(np.nanmin(finite_y)), float(np.nanmax(finite_y))
        y_range = max(y_hi - y_lo, 0.1)
        y_lo_shared = y_lo - 0.04 * y_range
        y_hi_shared = y_hi + 0.04 * y_range
    else:
        y_lo_shared, y_hi_shared = -1.0, 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    sc = None
    for plot_idx, (path, label, df) in enumerate(panel_data):
        ax = axes_flat[plot_idx]

        x = df["obs_log2"].values
        y = df["pred_log2"].values
        dist = df["enh_dist"].astype(float).values
        finite = np.isfinite(x) & np.isfinite(y)
        x_f, y_f, d_f = x[finite], y[finite], dist[finite]

        sc = ax.scatter(x_f, y_f, s=20, alpha=0.85, c=d_f, cmap=cmap,
                        norm=norm, edgecolor="none", zorder=3)

        if len(x_f) >= 2 and np.std(x_f) > 1e-12:
            m, b = np.polyfit(x_f, y_f, deg=1)
            xlo, xhi = x_f.min(), x_f.max()
            ax.plot([xlo, xhi], [m * xlo + b, m * xhi + b],
                    color=pal[1], linewidth=2.5, zorder=4)
            pr, pp = pearsonr(x_f, y_f)
            sr, sp = spearmanr(x_f, y_f)
            stats_txt = "Pearson r = {:.3f}\nSpearman ρ = {:.3f}\nn = {:d}".format(
                pr, sr, int(finite.sum()))
        else:
            stats_txt = "n = {:d}".format(int(finite.sum()))

        ax.text(
            0.03, 0.97, stats_txt,
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.85, edgecolor="none"),
        )
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel(
            "−log₂(1 − y_delta)\n(+ve = observed knockdown)", fontsize=11)
        if plot_idx == 0:
            ax.set_ylabel(
                "log₂(pred WT / pred CRISPRi)\n(+ve = predicted knockdown)", fontsize=11)
        else:
            ax.set_ylabel("")
        ax.set_title(label if label else "Model {}".format(plot_idx + 1), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lo_shared, y_hi_shared)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    if sc is not None:
        used = list(axes_flat[:n_models])
        cbar = fig.colorbar(sc, ax=used, shrink=0.85, pad=0.02, fraction=0.04)
        cbar.set_label("Distance to TSS (bp)", fontsize=11)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print("Log2 scatter saved to: {}".format(output_path))
    plt.close()


EFFECT_SCATTER_TRUNC_Y = -0.1


def plot_effect_scatter_panels(result_files, labels, output_path, dpi, max_enh_dist=None, min_enh_dist=None, max_width_subplots=3):
    """Scatter panels: x=experimental effect, y=predicted effect.

    Points coloured by ``enh_dist`` with a shared log-scale viridis colorbar across panels.
    Y-axis is shared across panels. Predictions below ``EFFECT_SCATTER_TRUNC_Y`` (-0.1)
    are plotted as ▼ markers at the floor; the *raw* values still feed the best-fit
    line and Pearson/Spearman r.
    """
    from matplotlib.colors import LogNorm, Normalize

    pal = setup_plot_style()

    panel_data = []
    all_dist = []
    all_y_raw = []
    for result_file, label in zip(result_files, labels):
        df = pd.read_csv(result_file, sep=",")
        df = filter_enh_dist(df, max_enh_dist, min_enh_dist, source_label=label or result_file)
        df = process_dataframe(df)
        panel_data.append((label, df))
        all_dist.append(df["enh_dist"].astype(float).values)
        all_y_raw.append(df["pred_delta_raw"].astype(float).values)

    n_models = len(panel_data)
    if n_models <= max_width_subplots:
        nrows, ncols = 1, n_models
        figsize = (6 * n_models, 5.5)
    else:
        ncols = max_width_subplots
        nrows = (n_models + ncols - 1) // ncols
        figsize = (18, 5.5 * nrows)

    dist_concat = np.concatenate(all_dist) if all_dist else np.array([1.0])
    finite_dist = dist_concat[np.isfinite(dist_concat) & (dist_concat > 0)]
    if finite_dist.size >= 2 and finite_dist.min() < finite_dist.max():
        norm = LogNorm(vmin=max(float(finite_dist.min()), 1.0),
                       vmax=float(finite_dist.max()))
    else:
        vmax = float(finite_dist.max()) if finite_dist.size else 1.0
        norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = plt.cm.viridis

    y_concat = np.concatenate(all_y_raw) if all_y_raw else np.array([0.0])
    finite_y = y_concat[np.isfinite(y_concat)]
    y_top = float(np.nanmax(finite_y)) if finite_y.size else 1.0
    y_top = max(y_top, EFFECT_SCATTER_TRUNC_Y + 0.2)
    y_range = y_top - EFFECT_SCATTER_TRUNC_Y
    y_lo_shared = EFFECT_SCATTER_TRUNC_Y - 0.04 * y_range
    y_hi_shared = y_top + 0.04 * y_range

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    sc = None
    for idx, (label, df) in enumerate(panel_data):
        ax = axes[idx]
        x = df["y_delta"].astype(float).values
        y_raw = df["pred_delta_raw"].astype(float).values
        dist = df["enh_dist"].astype(float).values

        trunc_mask = y_raw < EFFECT_SCATTER_TRUNC_Y
        y_display = np.where(trunc_mask, EFFECT_SCATTER_TRUNC_Y, y_raw)
        ok = ~trunc_mask

        if ok.any():
            sc_ok = ax.scatter(x[ok], y_display[ok], s=14, alpha=0.85,
                               c=dist[ok], cmap=cmap, norm=norm,
                               marker="o", edgecolor="none", zorder=3)
            sc = sc_ok
        if trunc_mask.any():
            sc_tr = ax.scatter(x[trunc_mask], y_display[trunc_mask], s=36, alpha=0.9,
                               c=dist[trunc_mask], cmap=cmap, norm=norm,
                               marker="v", edgecolor="none", zorder=4)
            if sc is None:
                sc = sc_tr

        if len(df) >= 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y_raw, deg=1)
            x_line = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(x_line, m * x_line + b, color=pal[1], linewidth=2.5, zorder=5)

        pear_r = pd.Series(x).corr(pd.Series(y_raw), method="pearson")
        spear_r = pd.Series(x).corr(pd.Series(y_raw), method="spearman")
        stats_txt = "Pearson r = {:.3f}\nSpearman r = {:.3f}\nn = {}".format(pear_r, spear_r, len(df))
        ax.text(
            0.03, 0.97, stats_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"),
        )

        ax.set_xlabel("Experimental effect")
        if idx == 0:
            ax.set_ylabel("Predicted effect")
        else:
            ax.set_ylabel("")
        ax.set_title(label if label else "Model {}".format(idx + 1), fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_lo_shared, y_hi_shared)

    for idx in range(n_models, len(axes)):
        axes[idx].axis("off")

    if sc is not None:
        used = list(axes[:n_models])
        cbar = fig.colorbar(sc, ax=used, shrink=0.85, pad=0.02, fraction=0.04)
        cbar.set_label("Distance to TSS (bp)", fontsize=11)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Effect scatter plot saved to: {output_path}")
    plt.close()


def plot_multiple_models(result_files, labels, output_path, dpi, y_scale='linear', max_enh_dist=None, min_enh_dist=None, max_width_subplots=3):
    """Create subplot comparison of multiple models."""
    pal = setup_plot_style()
    n_models = len(result_files)

    # Calculate subplot layout (prefer horizontal layout)
    if n_models <= max_width_subplots:
        nrows, ncols = 1, n_models
        figsize = (10 * n_models, 6)
    else:
        ncols = max_width_subplots
        nrows = (n_models + ncols - 1) // ncols  # Ceiling division
        figsize = (30, 6 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Ensure axes is always a flat array
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]
    
    # Plot each model without individual legends and y-labels
    for idx, (result_file, label) in enumerate(zip(result_files, labels)):
        df = pd.read_csv(result_file, sep=',')
        df = filter_enh_dist(df, max_enh_dist, min_enh_dist, source_label=label or result_file)
        df = process_dataframe(df)
        plot_single_model(df, axes[idx], pal, title=label, show_legend=False, show_ylabel=False, y_scale=y_scale)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Add a single y-axis label for all subplots
    fig.text(0.04, 0.5, 'Change in Expression (%)\ndue to Enhancer Knockdown', 
             va='center', rotation='vertical', fontsize=12)
    
    # Create a single legend for all subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=11, 
               bbox_to_anchor=(0.85, 0.5), loc='center left', frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.85)  # Make room for the y-label and legend
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_single(result_file, output_path, dpi, y_scale='linear', max_enh_dist=None, min_enh_dist=None):
    """Create plot for a single model."""
    pal = setup_plot_style()

    df = pd.read_csv(result_file, sep=',')
    df = filter_enh_dist(df, max_enh_dist, min_enh_dist, source_label=result_file)
    df = process_dataframe(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single_model(df, ax, pal, y_scale=y_scale)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


PEARSON_BIN_PALETTE = ["#A65141", "#E7CDC2", "#80A0C7", "#394165"]
DIST_BIN_EDGES_BP = [(0, 20_000), (20_000, 50_000), (50_000, 100_000),
                     (100_000, 250_000), (250_000, 500_000)]
DIST_BIN_LABELS = ["0–20k", "20–50k", "50–100k", "100–250k", "250–500k"]
MIN_BIN_N = 5  # minimum points to compute a meaningful Pearson r


def derive_pearson_bin_output_path(output_path):
    """Auto-generate companion path for the Pearson-r-by-distance-bin figure."""
    root, ext = os.path.splitext(output_path)
    return "{}_pearson_by_dist{}".format(root, ext or ".png")


def _pearson_per_bin(df, x_col="pred_delta_raw", y_col="y_delta", min_n=MIN_BIN_N):
    """Return [(bin_label, r_or_None, n)] for each bin in DIST_BIN_EDGES_BP.

    r is None when n<min_n or either column has zero variance (regression undefined).
    """
    from scipy.stats import pearsonr

    out = []
    dist = df["enh_dist"].astype(float).values
    for (lo, hi), label in zip(DIST_BIN_EDGES_BP, DIST_BIN_LABELS):
        mask = (dist >= lo) & (dist < hi)
        sub = df.loc[mask]
        if len(sub) < min_n:
            out.append((label, None, len(sub)))
            continue
        x = sub[x_col].astype(float).values
        y = sub[y_col].astype(float).values
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if len(x) < min_n or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            out.append((label, None, int(len(x))))
            continue
        r, _ = pearsonr(x, y)
        out.append((label, float(r), int(len(x))))
    return out


def plot_pearson_by_distance(
    result_files, labels, output_path, dpi,
    max_enh_dist=None, min_enh_dist=None,
    title=None
):
    """Grouped bar chart: Pearson r (y) vs enh_dist bin (x), one bar per model per bin.

    Bins (bp): 0–20k, 20–50k, 50–100k, 100–200k, 200–500k. Bins where a model has
    < MIN_BIN_N points (e.g. Enformer past ~100 kb) leave a visible gap. ``n=`` is
    annotated above each bar (below for negative r) to show CRE count per bin.
    Uses pred_delta_raw vs y_delta — same columns as plot_effect_scatter_panels.
    """
    setup_plot_style()
    pal = PEARSON_BIN_PALETTE

    model_results = []
    for path, label in zip(result_files, labels):
        df = pd.read_csv(path)
        if "enh_dist" not in df.columns:
            print("  [pearson by dist] skipping '{}' — no enh_dist column".format(label or path))
            continue
        if max_enh_dist is not None or min_enh_dist is not None:
            df = filter_enh_dist(df, max_enh_dist, min_enh_dist, source_label=label or path)
        df = process_dataframe(df)
        bin_results = _pearson_per_bin(df)
        model_results.append((label or os.path.basename(path), bin_results))
        bin_summary = ", ".join(
            "{}: r={}{}".format(
                lbl,
                "{:.3f}".format(r) if r is not None else "NA",
                " (n={})".format(n),
            )
            for lbl, r, n in bin_results
        )
        print("  [pearson by dist] {} → {}".format(label or path, bin_summary))

    if not model_results:
        print("  [pearson by dist] no eligible CSVs; skipping plot")
        return

    n_models = len(model_results)
    n_bins = len(DIST_BIN_LABELS)
    x_pos = np.arange(n_bins)
    bar_w = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(11, 2.0 * n_bins + 2), 6))

    all_r = [r for _, br in model_results for (_, r, _) in br if r is not None]
    pad = 0.08 if all_r else 0.0
    y_lo = min(all_r + [0.0]) - pad if all_r else -0.1
    y_hi = max(all_r + [0.0]) + pad if all_r else 0.1

    for m_idx, (label, bin_results) in enumerate(model_results):
        color = pal[m_idx % len(pal)]
        offset = (m_idx - (n_models - 1) / 2) * bar_w
        label_used = False
        for b_idx, (_, r_val, n) in enumerate(bin_results):
            x = x_pos[b_idx] + offset
            if r_val is None:
                # Show n= at y=0 so empty/sparse bins still report sample size
                ax.text(x, 0.0, "n={}".format(n), ha="center", va="bottom",
                        fontsize=8, color="grey", alpha=0.8)
                continue
            ax.bar(
                x, r_val, width=bar_w, color=color,
                edgecolor="white", linewidth=0.5,
                label=(label if not label_used else "_nolegend_"),
                zorder=3,
            )
            label_used = True
            text_y = r_val + 0.015 if r_val >= 0 else r_val - 0.015
            va = "bottom" if r_val >= 0 else "top"
            ax.text(x, text_y, "n={}".format(n), ha="center", va=va, fontsize=8)

    ax.axhline(0, color="grey", linewidth=0.8, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(DIST_BIN_LABELS)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Distance to TSS (bp)", fontsize=12)
    ax.set_ylabel("Pearson r", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, bbox_to_anchor=(1.02, 0.5),
              loc="center left", frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print("Pearson-by-distance plot saved to: {}".format(output_path))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot Gasperini benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--results',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to result CSV file(s). Multiple files will create subplots.'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=None,
        help='Labels for each model (optional). Must match number of result files.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for the plot (e.g., ./results/test_Gasperini_enformer/plots/gasperini_results.png)'
    )
    
    parser.add_argument(
        '--y_scale',
        type=str,
        default='linear',
        choices=['linear', 'log'],
        help=(
            'Y-axis scale (default: linear). '
            'With "log", non-positive y is floored for display where needed; truncated model '
            'points clipped to 0 (triangles) are plotted at y=0.001.'
        )
    )
    
    parser.add_argument(
        '--max_width_subplots',
        type=int,
        default=3,
        help='Maximum number of subplots in a row (default: 3)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=1000,
        help='DPI for the plot (default: 1000)'
    )
    
    parser.add_argument(
        '--max_enh_dist',
        type=int,
        default=None,
        metavar='BP',
        help=(
            'If set, keep only enhancer–gene pairs with enh_dist (bp) at most this value '
            '(column from benchmark CSVs). Example: 100000 for pairs within 100 kb of the TSS — '
            'useful for fair Enformer vs AlphaGenome comparison when receptive fields differ. '
            'Default: no filter (all pairs in each CSV).'
        )
    )

    parser.add_argument(
        '--min_enh_dist',
        type=int,
        default=None,
        metavar='BP',
        help=(
            'If set, keep only enhancer–gene pairs with enh_dist (bp) STRICTLY GREATER than this '
            'value. SequenceModelBenchmark uses > 990 to exclude near-promoter elements that may '
            'not be true distal enhancers. Default: no lower filter.'
        )
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.labels and len(args.labels) != len(args.results):
        raise ValueError(
            f"Number of labels ({len(args.labels)}) must match number of result files ({len(args.results)})"
        )

    missing = [p for p in args.results if not os.path.isfile(p)]
    if missing:
        lines = ["Result CSV file(s) not found:"]
        for p in missing:
            lines.append("  {}".format(os.path.abspath(p)))
        lines.append("")
        lines.append(
            "Run test_gasperini_alphagenome.py first, or pass the path printed when it finishes "
            "(e.g. .../Gasperini_AlphaGenome_base_results.csv). "
            "Default --save_path is ./results/gasperini_alphagenome/ unless you changed it."
        )
        raise FileNotFoundError("\n".join(lines))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate default labels if not provided
    if not args.labels:
        if len(args.results) == 1:
            args.labels = [None]
        else:
            args.labels = [f"Model {i+1}" for i in range(len(args.results))]
    
    # Create plot
    
    #ensure folder exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if len(args.results) == 1:
        plot_single(
            args.results[0],
            args.output,
            args.dpi,
            args.y_scale,
            max_enh_dist=args.max_enh_dist,
            min_enh_dist=args.min_enh_dist,
        )
    else:
        plot_multiple_models(
            args.results,
            args.labels,
            args.output,
            args.dpi,
            args.y_scale,
            max_enh_dist=args.max_enh_dist,
            min_enh_dist=args.min_enh_dist,
            max_width_subplots=args.max_width_subplots,
        )
    scatter_output = derive_scatter_output_path(args.output)
    plot_effect_scatter_panels(
        args.results,
        args.labels,
        scatter_output,
        args.dpi,
        max_enh_dist=args.max_enh_dist,
        min_enh_dist=args.min_enh_dist,
        max_width_subplots=args.max_width_subplots,
    )

    log2_output = derive_log2_output_path(args.output)
    plot_log2_scatter_panels(
        args.results,
        args.labels,
        log2_output,
        args.dpi,
        max_enh_dist=args.max_enh_dist,
        min_enh_dist=args.min_enh_dist,
        max_width_subplots=args.max_width_subplots,
    )

    log2_dist_output = derive_log2_dist_output_path(args.output)
    plot_log2_dist_panels(
        args.results,
        args.labels,
        log2_dist_output,
        args.dpi,
        max_enh_dist=args.max_enh_dist,
        min_enh_dist=args.min_enh_dist,
        max_width_subplots=args.max_width_subplots,
    )

    pearson_bin_output = derive_pearson_bin_output_path(args.output)
    plot_pearson_by_distance(
        args.results,
        args.labels,
        pearson_bin_output,
        args.dpi,
        max_enh_dist=args.max_enh_dist,
        min_enh_dist=args.min_enh_dist,
    )


if __name__ == '__main__':
    main()