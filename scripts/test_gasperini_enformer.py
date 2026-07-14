#!/usr/bin/env python3
"""
Test base Enformer on Gasperini et al. (2019) validated enhancer-gene pairs.

Evaluates model predictions on experimentally validated CRISPRi perturbations.
Compares wild-type predictions to dinucleotide-shuffled enhancer predictions
and correlates with experimental log fold changes.

Sequences are centred on the gene TSS (matching the SequenceModelBenchmark /
Avsec et al. approach); the enhancer is shuffled in place within this TSS-centred
window. CAGE readouts are aggregated at the TSS output bin, mapping the
input-frame ``tss_seq_index`` (from ``GasperiniDataset``) to the cropped output
via ``_tss_bin_from_seq_index`` (subtracting Enformer's TargetLengthCrop offset).

Base-model evaluation only. See README for adding new models.

Usage:
    python scripts/test_gasperini_enformer.py
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from enformer_pytorch import from_pretrained

from crispri_eval.datasets import GasperiniDataset


# Enformer (enformer-pytorch: EleutherAI/enformer-official-rough) constants
ENFORMER_SEQ_LEN = 196_608            # input bp
ENFORMER_BIN_BP = 128                 # output bin width (bp)
ENFORMER_NUM_BINS = 896               # cropped output length (TargetLengthCrop)
ENFORMER_CROP_BP_PER_SIDE = (ENFORMER_SEQ_LEN - ENFORMER_NUM_BINS * ENFORMER_BIN_BP) // 2  # 40960


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def _tss_bin_from_seq_index(tss_seq_index, num_bins=ENFORMER_NUM_BINS):
    """Map an input-sequence index (0..196607) to a cropped-output bin (0..895).

    Enformer's TargetLengthCrop drops 40,960 bp per side, so that crop must be
    subtracted before binning (else a TSS-centred window reads ~41 kb 3' of the TSS).
    """
    tss_output_pos = int(tss_seq_index) - ENFORMER_CROP_BP_PER_SIDE
    if tss_output_pos < 0 or tss_output_pos >= num_bins * ENFORMER_BIN_BP:
        raise ValueError(
            "TSS at sequence index {} maps to output position {}, outside Enformer's "
            "central crop [0, {}). Check the TSS-centred window and crop offset.".format(
                tss_seq_index, tss_output_pos, num_bins * ENFORMER_BIN_BP
            )
        )
    return int(tss_output_pos // ENFORMER_BIN_BP)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test base Enformer on Gasperini validated enhancer-gene pairs"
    )

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

    parser.add_argument(
        "--data_path",
        type=str,
        default="./metadata/",
        help="Path to directory containing Gasperini metadata files"
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
            "are excluded before inference to remove near-promoter elements. Default: 990."
        ),
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="./results/test_Gasperini_enformer/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for output files (default: auto-generated)"
    )

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
        "--high_confidence_subset",
        type=bool,
        default=True,
        help="Whether to use the high confidence subset of the dataset (default: True)"
    )

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
    if args.backbone_model_path is None:
        print("Loading Enformer from HuggingFace (EleutherAI/enformer-official-rough)...")
        model = from_pretrained("EleutherAI/enformer-official-rough")
    else:
        print(f"Loading Enformer from local path: {args.backbone_model_path}")
        model = from_pretrained(args.backbone_model_path)
    model.eval()
    return model


def get_target_indices(cell_line, assay):
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
    out = model(x)
    return out["human"]


def main():
    args = parse_args()

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

    print(f"\nLoading Gasperini dataset from {args.data_path}...")
    print("Sequence window: centred on gene TSS (matching SequenceModelBenchmark); enhancer shuffled in place.")

    dataset = GasperiniDataset(
        data_path=args.data_path,
        sequence_length=args.receptive_field,
        N=args.N,
        enhancer_bp=args.enhancer_bp,
        genome_build=args.genome_build,
        high_confidence_subset=args.high_confidence_subset,
        crispri_perturb_mode=args.crispri_perturb_mode,
        min_enh_dist=args.min_enh_dist,
    )

    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=None, num_workers=args.num_workers,
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
        y_delta = 1 - batch["y_delta"]

        with torch.no_grad():
            pred_wt = _enformer_predict(model, x_wt)
            bs = max(1, int(args.crispri_batch_size))
            n_c = x_crispri.shape[0]
            crispri_chunks = []
            for i in range(0, n_c, bs):
                crispri_chunks.append(_enformer_predict(model, x_crispri[i : i + bs]))
            pred_crispri = torch.cat(crispri_chunks, dim=0)

        pred_wt = pred_wt[:, :, model_inds]
        pred_crispri = pred_crispri[:, :, model_inds]

        # Map the TSS from input-sequence coordinates to the cropped output bin
        # (subtracting Enformer's TargetLengthCrop; see _tss_bin_from_seq_index).
        tss_idx = _tss_seq_index_from_batch(batch)
        n_bins = pred_wt.shape[1]
        tss_bin = _tss_bin_from_seq_index(tss_idx, num_bins=ENFORMER_NUM_BINS)
        bin_offset = args.num_central_bins // 2
        lo = max(0, tss_bin - bin_offset)
        hi = min(n_bins, tss_bin + bin_offset + 1)
        pred_wt = pred_wt[:, lo:hi, :]
        pred_crispri = pred_crispri[:, lo:hi, :]

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

    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(df_results["y_delta"], df_results["pred_delta"])
    spearman_r, spearman_p = spearmanr(df_results["y_delta"], df_results["pred_delta"])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Number of enhancer-gene pairs evaluated: {len(df_results)}")
    print(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
    print("=" * 80)

    if args.output_prefix is None:
        output_prefix = f"Gasperini_{model_name}"
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
        f.write("GASPERINI BENCHMARK SUMMARY\n")
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
        f.write(f"Central bins aggregated: {args.num_central_bins} (around TSS bin, TSS-centred window)\n")
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of pairs evaluated: {len(df_results)}\n")
        f.write(f"Pearson correlation: {pearson_r:.4f} (p={pearson_p:.2e})\n")
        f.write(f"Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})\n")
        f.write("=" * 80 + "\n")

    print(f"Summary saved to: {summary_path}")
    return df_results


if __name__ == "__main__":
    main()
