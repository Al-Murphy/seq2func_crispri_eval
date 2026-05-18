#!/usr/bin/env python3
"""
Test base Borzoi (PyTorch port: johahi/borzoi-pytorch) on Fulco et al. (2019)
CRISPRi enhancer knockdown pairs (Karollus et al. 2023 windowing via ziga tables).

Base-model evaluation only. See README for adding new models.

Borzoi includes RNA-Seq and CAGE tracks for K562 (erythroleukemia cell line),
making it a strong candidate for this CRISPRi perturbation benchmark.

  - Borzoi predicts at 32 bp bin resolution; output shape (B, 7611, 6144) for human.
  - The 6144 output bins cover the central 196,608 bp of the 524,288 bp input
    (crop = 163,840 bp on each side).
  - K562 RNA-Seq or CAGE tracks are looked up in calico/borzoi
    ``examples/targets_human.txt`` (column ``description``: ``RNA:K562`` or
    ``CAGE:...K562``).
  - Signal is summed over a TSS-centred bin window and selected K562 tracks.
  - Experimental labels match Enformer/AlphaGenome:
    ``y_delta = 1 - expr_ratio`` (higher = stronger repression vs WT).
  - Predicted delta:
    ``pred_delta = (pred_WT - pred_CRISPRi_mean) / (pred_WT + 1e-8)``.

  Sequence window: centred on the TSS from ``ziga_additional_columns.tsv``
  (Karollus et al.; ``main_tss_start`` / ``main_tss_end``). Strand for
  reverse-complement is taken from the Biomart ``--gene_tss_csv`` table (same
  format as Gasperini). The enhancer is shuffled in place within this window.

Usage:
    # Base model, RNA-Seq K562
    python scripts/test_fulco_borzoi.py \\
        --data_path ./metadata/fulco/ \\
        --gene_tss_csv ./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv

    # Base model, CAGE K562
    python scripts/test_fulco_borzoi.py --target_assay CAGE

    # Flashzoi backbone (faster on modern NVIDIA GPUs)
    python scripts/test_fulco_borzoi.py --use_flashzoi
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from crispri_eval.datasets import FulcoDataset
from crispri_eval.benchmark_utils import fulco_results_for_correlation


# Borzoi (johahi/borzoi-pytorch) constants
BORZOI_SEQ_LEN = 524_288              # input bp
BORZOI_BIN_BP = 32                    # output bin width (bp)
BORZOI_NUM_BINS = 6144                # default `bins_to_return` (return_center_bins_only=True)
BORZOI_CROP_BP_PER_SIDE = (BORZOI_SEQ_LEN - BORZOI_NUM_BINS * BORZOI_BIN_BP) // 2  # 163840
BORZOI_HUMAN_NUM_TRACKS = 7611
BORZOI_EMBED_CHANNELS = 1920

# Match Enformer/AlphaGenome Enformer-equivalent TSS window: 5 * 128 bp = 640 bp.
# At 32 bp Borzoi bins this is 20 bins -> 640 bp coverage.
DEFAULT_CENTRAL_BINS = 20


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def _tss_bin_from_seq_index(tss_seq_index, num_bins=BORZOI_NUM_BINS):
    """Map an input-sequence index (0..524287) to an output bin index (0..6143)."""
    tss_output_pos = int(tss_seq_index) - BORZOI_CROP_BP_PER_SIDE
    if tss_output_pos < 0 or tss_output_pos >= num_bins * BORZOI_BIN_BP:
        warnings.warn(
            "TSS at sequence index {} falls outside Borzoi's central crop "
            "[{}, {}); clamping the readout bin to the nearest edge.".format(
                tss_seq_index,
                BORZOI_CROP_BP_PER_SIDE,
                BORZOI_CROP_BP_PER_SIDE + num_bins * BORZOI_BIN_BP,
            )
        )
    bin_idx = tss_output_pos // BORZOI_BIN_BP
    return int(max(0, min(num_bins - 1, bin_idx)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Test base Borzoi (PyTorch) on Fulco CRISPRi benchmark (Karollus windows)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--backbone_replicate", type=str, default="johahi/borzoi-replicate-0",
        help=(
            "HuggingFace repo for the Borzoi backbone weights. "
            "Replicates: johahi/borzoi-replicate-{0,1,2,3}. "
            "Use --use_flashzoi to switch to johahi/flashzoi-replicate-{0,1,2,3}."
        ),
    )
    p.add_argument(
        "--use_flashzoi", action="store_true",
        help=(
            "Use Flashzoi (FlashAttention-2) backbone. Replaces "
            "'borzoi-replicate-*' with 'flashzoi-replicate-*' for the same fold index."
        ),
    )

    # -- Track selection ------------------------------------------------------
    p.add_argument(
        "--target_csv", type=str, default="./metadata/borzoi_human_targets.txt",
        help=(
            "Path to Borzoi targets_human.txt (from calico/borzoi). Tab-separated "
            "with columns: <index>, identifier, file, clip, clip_soft, scale, "
            "sum_stat, strand_pair, description. The first column carries the "
            "row index used as the model's track index."
        ),
    )
    p.add_argument(
        "--target_cell_line", type=str, default="K562",
        help="Cell line to match in the description column of borzoi_human_targets.txt.",
    )
    p.add_argument(
        "--target_assay", type=str, default="RNA", choices=["RNA", "CAGE"],
        help=(
            "Assay prefix in the description column: 'RNA' (RNA-Seq, ~94 K562 tracks) "
            "or 'CAGE' (4 K562 tracks). Borzoi reports both modalities on the same "
            "head; only the indices change."
        ),
    )
    p.add_argument(
        "--num_central_bins", type=int, default=DEFAULT_CENTRAL_BINS,
        help=(
            "Number of 32 bp bins to aggregate around the TSS bin. Default {} "
            "(= {} bp ≈ Enformer's 5 x 128 bp central window).".format(
                DEFAULT_CENTRAL_BINS, DEFAULT_CENTRAL_BINS * BORZOI_BIN_BP
            )
        ),
    )

    p.add_argument(
        "--shuffle_batch_size", type=int, default=2,
        help=(
            "Number of dinucleotide-shuffled CRISPRi sequences per forward pass. "
            "Lower if you hit CUDA OOM (524,288 bp inputs + 6144x7611 outputs are memory-heavy)."
        ),
    )

    # -- Data -----------------------------------------------------------------
    p.add_argument(
        "--data_path", type=str, default="./metadata/fulco/",
        help="Directory with ziga_additional_columns.tsv and enhancer_knockdown_effects.tsv.",
    )
    p.add_argument(
        "--gene_tss_csv", type=str,
        default="./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv",
        help="Biomart CSV with Gene stable ID and Strand (1 / -1); same format as Gasperini TSS export.",
    )
    p.add_argument("--genome_build", type=str, default="hg38", choices=["hg38", "hg19"])
    p.add_argument("--N", type=int, default=50,
                   help="Number of dinucleotide shuffles per enhancer.")
    p.add_argument("--enhancer_bp", type=int, default=2000,
                   help="Size of enhancer region to shuffle (bp).")
    p.add_argument(
        "--crispri_perturb_mode", type=str, default="dinucleotide",
        choices=["dinucleotide", "shuffle_bases", "random_permute"],
        help=(
            "CRISPRi negatives: dinucleotide (default); shuffle_bases — same bases in the "
            "enhancer, order uniformly shuffled (alias: random_permute)."
        ),
    )
    p.add_argument(
        "--validated_only",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="If True, keep only validated==True rows in ziga (default: True).",
    )
    p.add_argument(
        "--min_enh_dist", type=int, default=990, metavar="BP",
        help=(
            "Minimum enhancer-TSS distance (bp, exclusive). Pairs with enh_dist <= this "
            "value are excluded. SequenceModelBenchmark uses > 990 bp."
        ),
    )
    p.add_argument(
        "--fulco_corr_observed_subset", type=str, default="knockdown_only",
        choices=["all", "knockdown_only"],
        help=(
            "Rows for Pearson/Spearman vs pred_delta: 'all' — every evaluated pair; "
            "'knockdown_only' — y_delta > 0 only (measured RNA decrease)."
        ),
    )
    p.add_argument(
        "--gene", type=str, default=None,
        help="Optional: restrict evaluation to pairs for this 'target_gene_short'.",
    )

    # -- Output ---------------------------------------------------------------
    p.add_argument(
        "--save_path", type=str, default="./results/test_Fulco_borzoi/",
        help="Output directory.",
    )
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument(
        "--debug_export", action="store_true",
        help="Include per-pair diagnostics (track count, WT/shuffle spread) in the CSV.",
    )

    # -- Runtime --------------------------------------------------------------
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--autocast_dtype", type=str, default=None,
        choices=[None, "bf16", "fp16"],
        help=(
            "Run forward passes under torch.autocast(dtype=...) for speed. "
            "Recommended when --use_flashzoi (which requires modern NVIDIA GPUs)."
        ),
    )

    args = p.parse_args()

    # Flashzoi's FlashAttention layers require fp16/bf16 inputs.
    if args.use_flashzoi and args.autocast_dtype is None:
        print("--use_flashzoi: defaulting --autocast_dtype to bf16 (required by FlashAttention).")
        args.autocast_dtype = "bf16"

    return args


# ---------------------------------------------------------------------------
# Borzoi targets_human.txt -> K562 track indices
# ---------------------------------------------------------------------------

def get_borzoi_track_indices(target_csv, cell_line, assay):
    """
    Return (list of track indices, summary string) for tracks matching
    *cell_line* and *assay* in the borzoi_human_targets.txt description column.

    The first (unnamed) column of borzoi_human_targets.txt is the row index used
    as the model's output track index. ``assay`` is matched as a prefix of the
    ``description`` value (e.g. ``"RNA:"`` or ``"CAGE:"``).
    """
    if not os.path.exists(target_csv):
        raise FileNotFoundError(
            "Borzoi targets file not found: {}\n"
            "Download from calico/borzoi (examples/targets_human.txt) and pass "
            "via --target_csv.".format(target_csv)
        )

    df = pd.read_csv(target_csv, sep="\t")
    first_col = df.columns[0]
    if first_col in ("", "Unnamed: 0"):
        df = df.rename(columns={first_col: "track_index"})
    else:
        df["track_index"] = df.index

    desc_prefix = "{}:".format(assay)
    mask = (
        df["description"].astype(str).str.startswith(desc_prefix)
        & df["description"].astype(str).str.contains(cell_line, case=False, na=False)
    )
    matched = df[mask]
    if matched.empty:
        raise ValueError(
            "No tracks matched assay='{}' cell_line='{}' in {}.".format(
                assay, cell_line, target_csv
            )
        )

    indices = matched["track_index"].astype(int).tolist()
    descs = matched["description"].astype(str).tolist()
    extra = " (+{} more)".format(len(indices) - 4) if len(indices) > 4 else ""
    desc_summary = "{} {} {} tracks: {}{}".format(
        len(indices), cell_line, assay, ", ".join(descs[:4]), extra
    )
    return indices, desc_summary


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resolve_borzoi_repo(replicate_name, use_flashzoi):
    """Resolve the HuggingFace repo name, swapping in flashzoi if requested."""
    if use_flashzoi and "flashzoi" not in replicate_name:
        return replicate_name.replace("borzoi-replicate-", "flashzoi-replicate-")
    return replicate_name


def load_base_borzoi(replicate_name, use_flashzoi, device):
    """Load pretrained Borzoi via johahi/borzoi-pytorch."""
    from borzoi_pytorch import Borzoi

    repo = _resolve_borzoi_repo(replicate_name, use_flashzoi)
    print("Loading pretrained Borzoi from HuggingFace: {}".format(repo))
    model = Borzoi.from_pretrained(repo)
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _to_borzoi_input(x):
    """Convert (B, L, 4) -> (B, 4, L), as expected by Borzoi (channels-first)."""
    return x.permute(0, 2, 1).contiguous()


def _aggregate_predictions(pred, track_inds, tss_bin, central_bins):
    """
    Aggregate a Borzoi output (B, C, 6144) over a track subset and a bin window
    centred on ``tss_bin``. Returns a list of B scalars (sum over tracks and bins).
    """
    if track_inds is not None:
        pred = pred[:, track_inds, :]
    n_bins = pred.shape[-1]
    bin_offset = central_bins // 2
    lo = max(0, tss_bin - bin_offset)
    hi = min(n_bins, tss_bin + bin_offset + 1)
    pred_central = pred[:, :, lo:hi]
    return pred_central.sum(dim=(1, 2)).tolist()


def _autocast_ctx(device, autocast_dtype):
    if autocast_dtype is None or device == "cpu":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if autocast_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _predict_base(model, x, track_inds, tss_bin, central_bins, device, autocast_dtype):
    """Forward x through pretrained Borzoi and aggregate to a scalar per batch row."""
    x_dev = _to_borzoi_input(x).to(device)
    with torch.no_grad(), _autocast_ctx(device, autocast_dtype):
        out = model(x_dev, is_human=True)  # (B, 7611, 6144)
    return _aggregate_predictions(out.float(), track_inds, tss_bin, central_bins)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    os.makedirs(args.save_path, exist_ok=True)

    # ---- Model loading -------------------------------------------------------
    model = load_base_borzoi(args.backbone_replicate, args.use_flashzoi, device)
    model_name = "base_borzoi{}".format("_flashed" if args.use_flashzoi else "")
    print("Model name: {}".format(model_name))

    # ---- Track selection -----------------------------------------------------
    track_inds, track_desc = get_borzoi_track_indices(
        args.target_csv, args.target_cell_line, args.target_assay
    )
    print("Track selection ({}): {}".format(args.target_assay, track_desc))

    track_inds_tensor = torch.as_tensor(track_inds, dtype=torch.long, device=device)

    print("Aggregation window: {} bins x {} bp = {} bp".format(
        args.num_central_bins, BORZOI_BIN_BP, args.num_central_bins * BORZOI_BIN_BP
    ))

    # ---- Dataset -------------------------------------------------------------
    print("\nLoading Fulco dataset from {}...".format(args.data_path))
    print("Sequence window: centred on ziga TSS (Karollus et al.); enhancer shuffled in place.")
    dataset = FulcoDataset(
        data_path=args.data_path,
        sequence_length=BORZOI_SEQ_LEN,
        N=args.N,
        enhancer_bp=args.enhancer_bp,
        genome_build=args.genome_build,
        validated_only=args.validated_only,
        crispri_perturb_mode=args.crispri_perturb_mode,
        min_enh_dist=args.min_enh_dist,
        gene_tss_csv=args.gene_tss_csv,
        observed_subset=args.fulco_corr_observed_subset,
    )

    if args.gene is not None:
        before = len(dataset.crispri_data)
        dataset.crispri_data = dataset.crispri_data[
            dataset.crispri_data["target_gene_short"] == args.gene
        ].reset_index(drop=True)
        after = len(dataset.crispri_data)
        print("Filtered to gene '{}': {} -> {} pairs".format(args.gene, before, after))
        if after == 0:
            raise ValueError(
                "No Fulco pairs found for gene '{}'.".format(args.gene)
            )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,  # pre-batched
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # ---- Inference -----------------------------------------------------------
    print("\nRunning inference on {} enhancer-gene pairs...".format(len(dataset)))
    print("  {} dinucleotide shuffles per pair".format(args.N))
    print("  Context: {:,} bp  |  shuffle batch size: {}".format(
        BORZOI_SEQ_LEN, args.shuffle_batch_size
    ))

    all_y_deltas = []
    all_pred_deltas = []
    all_pred_wt = []
    all_pred_crispri_mean = []
    all_enh_dists = []
    all_gene_names = []
    all_gene_ids = []
    all_enh_locs = []
    all_tss = []
    all_strands = []
    all_tss_seq_index = []
    all_tss_bin = []
    all_debug_track_count = []
    all_debug_crispri_std = []
    all_debug_crispri_min = []
    all_debug_crispri_max = []

    def _predict_batch(x_batch, tss_bin):
        return _predict_base(
            model, x_batch, track_inds_tensor, tss_bin, args.num_central_bins,
            device, args.autocast_dtype,
        )

    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue

        x_wt = batch["x_WT"]
        x_crispri = batch["x_crispri"]

        y_delta = (1.0 - batch["y_delta"]).item()

        tss_seq_idx = _tss_seq_index_from_batch(batch)
        tss_bin = _tss_bin_from_seq_index(tss_seq_idx, num_bins=BORZOI_NUM_BINS)

        pred_wt = _predict_batch(x_wt, tss_bin)[0]

        pred_crispri_list = []
        bs = max(1, int(args.shuffle_batch_size))
        for i in range(0, x_crispri.shape[0], bs):
            pred_crispri_list.extend(_predict_batch(x_crispri[i : i + bs], tss_bin))

        pred_crispri_mean = sum(pred_crispri_list) / len(pred_crispri_list)
        pred_crispri_std = float(np.std(pred_crispri_list))
        pred_crispri_min = float(np.min(pred_crispri_list))
        pred_crispri_max = float(np.max(pred_crispri_list))

        pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)

        all_y_deltas.append(y_delta)
        all_pred_deltas.append(pred_delta)
        all_pred_wt.append(float(pred_wt))
        all_pred_crispri_mean.append(float(pred_crispri_mean))
        all_enh_dists.append(batch["enh_dist"])
        all_gene_names.append(batch["gene_name"])
        all_gene_ids.append(batch["gene_id"])
        all_enh_locs.append(batch["enh_loc"])
        all_tss.append(batch["tss"])
        all_strands.append(batch["strand"])
        all_tss_seq_index.append(tss_seq_idx)
        all_tss_bin.append(tss_bin)
        all_debug_track_count.append(len(track_inds))
        all_debug_crispri_std.append(pred_crispri_std)
        all_debug_crispri_min.append(pred_crispri_min)
        all_debug_crispri_max.append(pred_crispri_max)

    if len(all_y_deltas) == 0:
        print("No pairs evaluated -- exiting.")
        return None

    # ---- Results -------------------------------------------------------------
    print("\nCompiling results...")
    df_results = pd.DataFrame({
        "gene_id":           all_gene_ids,
        "gene_name":         all_gene_names,
        "enh_loc":           all_enh_locs,
        "tss":               all_tss,
        "tss_seq_index":     all_tss_seq_index,
        "tss_bin":           all_tss_bin,
        "strand":            all_strands,
        "enh_dist":          all_enh_dists,
        "y_delta":           all_y_deltas,
        "pred_delta":        all_pred_deltas,
        "pred_wt":           all_pred_wt,
        "pred_crispri_mean": all_pred_crispri_mean,
    })
    if args.debug_export:
        df_results["debug_track_count_used"] = all_debug_track_count
        df_results["debug_crispri_std"] = all_debug_crispri_std
        df_results["debug_crispri_min"] = all_debug_crispri_min
        df_results["debug_crispri_max"] = all_debug_crispri_max

    n_full = len(df_results)
    df_corr = fulco_results_for_correlation(df_results, args.fulco_corr_observed_subset)
    n_corr = len(df_corr)
    if n_corr < 2:
        raise ValueError(
            "Need at least 2 pairs for correlation after --fulco_corr_observed_subset={!r}; "
            "got {} / {} total.".format(args.fulco_corr_observed_subset, n_corr, n_full)
        )

    pearson_r, pearson_p = pearsonr(df_corr["y_delta"], df_corr["pred_delta"])
    spearman_r, spearman_p = spearmanr(df_corr["y_delta"], df_corr["pred_delta"])

    # Log2 knockdown strengths (Karollus-style sign).
    _eps = 1e-12
    expr_ratio = np.clip(1.0 - df_corr["y_delta"].astype(float).values, _eps, None)
    obs_log_strength = -np.log2(expr_ratio)
    wt = np.maximum(df_corr["pred_wt"].astype(float).values, _eps)
    cr = np.maximum(df_corr["pred_crispri_mean"].astype(float).values, _eps)
    pred_log_strength = np.log2(wt / cr)
    log_pear_r, log_pear_p = pearsonr(obs_log_strength, pred_log_strength)
    log_spr_r, log_spr_p = spearmanr(obs_log_strength, pred_log_strength)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("Number of enhancer-gene pairs evaluated: {}".format(n_full))
    print("Pairs used for correlations (--fulco_corr_observed_subset={}): {}".format(
        args.fulco_corr_observed_subset, n_corr
    ))
    print("Pearson correlation (linear y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(
        pearson_r, pearson_p
    ))
    print("Spearman correlation (linear):                       {:.4f}  (p={:.2e})".format(
        spearman_r, spearman_p
    ))
    print("Log2 knockdown alignment (-log2(expr_ratio) vs log2(pred_wt/pred_crispri)):")
    print("  Pearson  r: {:.4f}  (p={:.2e})".format(log_pear_r, log_pear_p))
    print("  Spearman r: {:.4f}  (p={:.2e})".format(log_spr_r, log_spr_p))
    print("=" * 80)

    # ---- Save ----------------------------------------------------------------
    if args.output_prefix is None:
        flash_tag = "_flashzoi" if args.use_flashzoi else ""
        assay_tag = "_{}".format(args.target_assay.lower())
        output_prefix = "Fulco_Borzoi_base{}{}".format(flash_tag, assay_tag)
    else:
        output_prefix = args.output_prefix

    csv_path = os.path.join(args.save_path, "{}_results.csv".format(output_prefix))
    df_results.to_csv(csv_path, index=False)
    print("\nResults saved to: {}".format(csv_path))

    summary_path = os.path.join(args.save_path, "{}_summary.txt".format(output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULCO BENCHMARK SUMMARY -- Borzoi (PyTorch)\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write("Model:             base Borzoi\n")
        f.write("Backbone replicate: {}\n".format(
            _resolve_borzoi_repo(args.backbone_replicate, args.use_flashzoi)
        ))
        f.write("Flashzoi:          {}\n".format(args.use_flashzoi))
        f.write("\nEVALUATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("Context window:    {:,} bp\n".format(BORZOI_SEQ_LEN))
        f.write("Bin resolution:    {} bp ({} output bins)\n".format(BORZOI_BIN_BP, BORZOI_NUM_BINS))
        f.write("Central crop:      [{}, {}) bp of input window\n".format(
            BORZOI_CROP_BP_PER_SIDE, BORZOI_SEQ_LEN - BORZOI_CROP_BP_PER_SIDE
        ))
        f.write("Genome build:      {}\n".format(args.genome_build))
        f.write("Fulco data path:   {}\n".format(args.data_path))
        f.write("Gene TSS CSV:      {}\n".format(args.gene_tss_csv))
        f.write("Validated only:    {}\n".format(args.validated_only))
        f.write("N shuffles:        {}\n".format(args.N))
        f.write("Enhancer window:   {} bp\n".format(args.enhancer_bp))
        f.write("Target cell line:  {}\n".format(args.target_cell_line))
        f.write("Target assay:      {}\n".format(args.target_assay))
        f.write("Tracks:            {}\n".format(track_desc))
        f.write("Aggregation:       sum over {} central bins ({} bp around TSS bin)\n".format(
            args.num_central_bins, args.num_central_bins * BORZOI_BIN_BP
        ))
        f.write("Scoring:           fractional drop (pred_WT - pred_CRISPRi) / (pred_WT + eps)\n")
        f.write("Fulco subset:      {}\n".format(args.fulco_corr_observed_subset))
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Pairs evaluated:                 {}\n".format(n_full))
        f.write("Pairs used for Pearson/Spearman: {}\n".format(n_corr))
        f.write("Pearson r  (linear):  {:.4f}  (p={:.2e})\n".format(pearson_r, pearson_p))
        f.write("Spearman r (linear):  {:.4f}  (p={:.2e})\n".format(spearman_r, spearman_p))
        f.write("Pearson r  (log2):    {:.4f}  (p={:.2e})\n".format(log_pear_r, log_pear_p))
        f.write("Spearman r (log2):    {:.4f}  (p={:.2e})\n".format(log_spr_r, log_spr_p))
        f.write("=" * 80 + "\n")

    print("Summary saved to: {}".format(summary_path))
    return df_results


if __name__ == "__main__":
    main()
