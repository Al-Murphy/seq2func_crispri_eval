#!/usr/bin/env python3
"""
Test base Enformer on Fulco et al. (2019) CRISPRi enhancer knockdown pairs.

Uses the Avsec / Karollus et al. (2023) window definitions (``ziga_additional_columns.tsv``)
plus ``enhancer_knockdown_effects.tsv`` from the Zenodo ``SequenceBenchmark.zip``
(``Data/Fulco_CRISPRi/``). Sequences are centred on the TSS in ``main_tss_start`` /
``main_tss_end`` (Karollus pipeline), with the enhancer shuffled in place — same
protocol as ``test_gasperini_enformer.py``.

Base-model evaluation only. See README for adding new models.

Pearson/Spearman rows are chosen with ``--fulco_corr_observed_subset`` (default
``knockdown_only``, matching ``plot_fulco_results.py --observed_mode decrease_only``).
Use ``all`` to include every evaluated pair (matches ``--observed_mode signed``).

Data setup (copy from Zenodo after extracting ``SequenceBenchmark.zip``):
    metadata/fulco/ziga_additional_columns.tsv
    metadata/fulco/enhancer_knockdown_effects.tsv

Strand (Biomart, same format as Gasperini TSS table):
    --gene_tss_csv ./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv

Usage:
    python scripts/test_fulco_enformer.py \\
        --data_path ./metadata/fulco/ \\
        --gene_tss_csv ./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from enformer_pytorch import from_pretrained

from crispri_eval.datasets import FulcoDataset
from crispri_eval.benchmark_utils import fulco_results_for_correlation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test base Enformer on Fulco CRISPRi enhancer-gene pairs (Karollus windows)"
    )

    # Model configuration
    parser.add_argument(
        "--backbone_model_path",
        type=str,
        default=None,
        help=(
            "Path to a local Enformer weights file. If None (default), loads the "
            "official HuggingFace checkpoint via "
            "enformer_pytorch.from_pretrained('EleutherAI/enformer-official-rough')."
        ),
    )

    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="./metadata/fulco/",
        help="Directory with ziga_additional_columns.tsv and enhancer_knockdown_effects.tsv"
    )
    parser.add_argument(
        "--gene_tss_csv",
        type=str,
        default="./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv",
        help=(
            "Biomart-style CSV with Gene stable ID and Strand (1 / -1), same columns as "
            "Gasperini gene TSS export. Used to orient sequences for − strand genes."
        ),
    )
    parser.add_argument(
        "--receptive_field",
        type=int,
        default=196_608,
        help="Model receptive field in base pairs (default: 196608 for Enformer)"
    )
    parser.add_argument(
        "--genome_build",
        type=str,
        default="hg38",
        choices=["hg38", "hg19"],
        help="Genome build to use"
    )

    # CRISPRi simulation parameters
    parser.add_argument(
        "--N",
        type=int,
        default=50,
        help="Number of dinucleotide shuffles per enhancer (default: 50)"
    )
    parser.add_argument(
        "--enhancer_bp",
        type=int,
        default=2000,
        help="Size of enhancer region to shuffle in bp (default: 2000)"
    )
    parser.add_argument(
        "--crispri_perturb_mode",
        type=str,
        default="dinucleotide",
        choices=["dinucleotide", "shuffle_bases"],
        help=(
            "CRISPRi negatives: dinucleotide (default, preserves dinucleotide frequencies); "
            "shuffle_bases — same bases as WT in the enhancer window, order uniformly shuffled "
        ),
    )
    parser.add_argument(
        "--min_enh_dist",
        type=int,
        default=990,
        metavar="BP",
        help=(
            "Minimum enhancer–TSS distance (bp, exclusive). Pairs with enh_dist <= this value "
            "are excluded before inference to remove near-promoter elements. "
            "SequenceModelBenchmark uses > 990 bp. Default: 990."
        ),
    )
    parser.add_argument(
        "--fulco_corr_observed_subset",
        type=str,
        default="knockdown_only",
        choices=["all", "knockdown_only"],
        help=(
            "Rows for Pearson/Spearman vs pred_delta: 'all' — every evaluated pair (same as "
            "plot_fulco_results.py --observed_mode signed, includes measured RNA increases); "
            "'knockdown_only' — y_delta > 0 only (same as plot --observed_mode decrease_only)."
        ),
    )

    # Output configuration
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/test_Fulco_enformer/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: auto-generated)"
    )

    # Evaluation configuration
    parser.add_argument(
        "--target_cell_line",
        type=str,
        default="K562",
        help="Cell line to evaluate (must match available CAGE tracks)"
    )
    parser.add_argument(
        "--target_assay",
        type=str,
        default="CAGE",
        help="Assay type to evaluate"
    )
    parser.add_argument(
        "--num_central_bins",
        type=int,
        default=5,
        help="Number of central bins to aggregate around TSS (default: 5 = -2 to +2)"
    )
    parser.add_argument(
        "--validated_only",
        type=bool,
        default=True,
        help="If True, keep only rows with validated==True in ziga (default: True)"
    )

    # Runtime configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--crispri_batch_size",
        type=int,
        default=25,
        help=(
            "How many CRISPRi (shuffled) sequences to run per forward pass. "
            "Lower this if you increase --N and hit CUDA OOM (default: 25)."
        ),
    )

    return parser.parse_args()


def load_model(args):
    """Load base pretrained Enformer."""
    if args.backbone_model_path is None:
        print("Loading Enformer from HuggingFace (EleutherAI/enformer-official-rough)...")
        model = from_pretrained("EleutherAI/enformer-official-rough")
    else:
        print(f"Loading Enformer from local path: {args.backbone_model_path}")
        model = from_pretrained(args.backbone_model_path)
    model.eval()
    return model


def get_target_indices(cell_line, assay):
    """Get model output indices for specific cell line and assay."""
    targets_url = "https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt"
    df_targets = pd.read_csv(targets_url, sep="\t")

    model_inds = df_targets[
        (df_targets["description"].str.contains(assay, case=False)) &
        (df_targets["description"].str.contains(cell_line, case=False))
    ]["index"].tolist()

    if len(model_inds) == 0:
        raise ValueError(f"No {assay} tracks found for {cell_line}")

    print(f"Found {len(model_inds)} {assay} track(s) for {cell_line}")
    return model_inds


def _enformer_predict(model, x):
    """Forward pass; base Enformer returns a dict with ``human`` head."""
    out = model(x)
    return out["human"]


def main():
    args = parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    model_name = "base_enformer"

    print(f"\n{'='*80}")
    print("Model Configuration:")
    print(f"  Type: base")
    print(f"  Name: {model_name}")
    print(f"{'='*80}\n")

    os.makedirs(args.save_path, exist_ok=True)

    model = load_model(args).to(device)

    print(f"\nLoading Fulco dataset from {args.data_path}...")
    print("Sequence window: centred on TSS from ziga (Karollus et al.); enhancer shuffled in place.")

    dataset = FulcoDataset(
        data_path=args.data_path,
        sequence_length=args.receptive_field,
        N=args.N,
        enhancer_bp=args.enhancer_bp,
        genome_build=args.genome_build,
        validated_only=args.validated_only,
        crispri_perturb_mode=args.crispri_perturb_mode,
        min_enh_dist=args.min_enh_dist,
        gene_tss_csv=args.gene_tss_csv,
        observed_subset=args.fulco_corr_observed_subset,
    )

    # batch_size=None for pre-batched samples
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,
        num_workers=args.num_workers,
    )

    print(f"\nFinding {args.target_assay} tracks for {args.target_cell_line}...")
    model_inds = get_target_indices(args.target_cell_line, args.target_assay)

    print(f"\nRunning inference on {len(dataset)} enhancer-gene pairs...")
    print(
        f"Each pair tested with {args.N} CRISPRi negatives "
        f"(crispri_perturb_mode={args.crispri_perturb_mode})"
    )
    print(f"CRISPRi forward batch size: {args.crispri_batch_size} (set --crispri_batch_size if OOM)")

    all_y_deltas = []
    all_pred_deltas = []
    all_enh_dists = []
    all_gene_names = []
    all_gene_ids = []
    all_enh_locs = []
    all_tss = []
    all_strands = []
    all_tss_seq_index = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        x_crispri = batch["x_crispri"].to(device)
        x_wt = batch["x_WT"].to(device)
        # Convert Fulco fraction change to fraction-remaining → delta
        y_delta = 1 - batch["y_delta"]

        # Run model: WT once; CRISPRi shuffles in chunks to avoid OOM when N is large
        with torch.no_grad():
            pred_wt = _enformer_predict(model, x_wt)
            bs = max(1, int(args.crispri_batch_size))
            n_c = x_crispri.shape[0]
            crispri_chunks = []
            for i in range(0, n_c, bs):
                crispri_chunks.append(_enformer_predict(model, x_crispri[i : i + bs]))
            pred_crispri = torch.cat(crispri_chunks, dim=0)

        # Filter to target tracks
        pred_wt = pred_wt[:, :, model_inds]
        pred_crispri = pred_crispri[:, :, model_inds]

        # Bins around TSS (window is TSS-centred; TSS bin is at/near centre)
        tss_idx = batch["tss_seq_index"]
        tss_idx = int(tss_idx.item()) if torch.is_tensor(tss_idx) else int(tss_idx)
        n_bins = pred_wt.shape[1]
        tss_bin = tss_idx // 128
        tss_bin = max(0, min(n_bins - 1, tss_bin))
        bin_offset = args.num_central_bins // 2
        lo = max(0, tss_bin - bin_offset)
        hi = min(n_bins, tss_bin + bin_offset + 1)
        pred_wt = pred_wt[:, lo:hi, :]
        pred_crispri = pred_crispri[:, lo:hi, :]

        # Aggregate: sum across bins and tracks, average across shuffles
        pred_wt_agg = pred_wt.sum(dim=(1, 2))[0]
        pred_crispri_agg = pred_crispri.sum(dim=(1, 2)).mean(dim=0)

        pred_delta = (pred_wt_agg - pred_crispri_agg) / (pred_wt_agg + 1e-8)

        all_y_deltas.append(y_delta.item())
        all_pred_deltas.append(pred_delta.cpu().item())
        all_enh_dists.append(batch["enh_dist"])
        all_gene_names.append(batch["gene_name"])
        all_gene_ids.append(batch["gene_id"])
        all_enh_locs.append(batch["enh_loc"])
        all_tss.append(batch["tss"])
        all_strands.append(batch["strand"])
        all_tss_seq_index.append(tss_idx)

    print("\nCompiling results...")
    df_results = pd.DataFrame({
        "gene_id": all_gene_ids,
        "gene_name": all_gene_names,
        "enh_loc": all_enh_locs,
        "tss": all_tss,
        "tss_seq_index": all_tss_seq_index,
        "strand": all_strands,
        "enh_dist": all_enh_dists,
        "y_delta": all_y_deltas,
        "pred_delta": all_pred_deltas,
    })

    n_full = len(df_results)
    df_corr = fulco_results_for_correlation(df_results, args.fulco_corr_observed_subset)
    n_corr = len(df_corr)
    if n_corr < 2:
        raise ValueError(
            "Need at least 2 pairs for correlation after --fulco_corr_observed_subset={!r}; "
            "got {} / {} total.".format(args.fulco_corr_observed_subset, n_corr, n_full)
        )

    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(df_corr["y_delta"], df_corr["pred_delta"])
    spearman_r, spearman_p = spearmanr(df_corr["y_delta"], df_corr["pred_delta"])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Number of enhancer-gene pairs evaluated: {n_full}")
    print(
        "Pairs used for Pearson/Spearman (--fulco_corr_observed_subset={}): {}".format(
            args.fulco_corr_observed_subset, n_corr
        )
    )
    print(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    print("=" * 80)

    if args.output_prefix is None:
        output_prefix = f"Fulco_{model_name}"
    else:
        output_prefix = args.output_prefix
    if args.crispri_perturb_mode == "shuffle_bases":
        output_prefix = f"{output_prefix}_shuffle_bases"

    output_csv = os.path.join(args.save_path, f"{output_prefix}_results.csv")
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")

    summary_path = os.path.join(args.save_path, f"{output_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULCO BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: base Enformer\n")
        f.write(f"Backbone path: {args.backbone_model_path or 'HuggingFace EleutherAI/enformer-official-rough'}\n")
        f.write("\nEVALUATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Receptive field: {args.receptive_field} bp\n")
        f.write(f"Genome build: {args.genome_build}\n")
        f.write(f"Target cell line: {args.target_cell_line}\n")
        f.write(f"Target assay: {args.target_assay}\n")
        f.write(f"Number of CRISPRi negatives per pair: {args.N}\n")
        f.write(f"CRISPRi forward batch size: {args.crispri_batch_size}\n")
        f.write(f"CRISPRi perturb mode: {args.crispri_perturb_mode}\n")
        f.write(f"Enhancer region size: {args.enhancer_bp} bp\n")
        f.write(f"Validated-only subset: {args.validated_only}\n")
        f.write(f"Gene TSS / strand file: {args.gene_tss_csv}\n")
        f.write(f"Central bins aggregated: {args.num_central_bins} (around TSS bin, TSS-centred window)\n")
        f.write(
            "Fulco correlation observed subset: {} (see --fulco_corr_observed_subset)\n".format(
                args.fulco_corr_observed_subset
            )
        )
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of pairs evaluated: {n_full}\n")
        f.write(f"Pairs used for Pearson/Spearman: {n_corr}\n")
        f.write(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})\n")
        f.write(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})\n")
        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {summary_path}")
    return df_results


if __name__ == "__main__":
    main()
