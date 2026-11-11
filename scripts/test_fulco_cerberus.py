#!/usr/bin/env python3
"""
Test Cerberus (Calico; Kelley et al. 2026, calico/baskerville) on Fulco et al.
(2019) CRISPRi enhancer knockdown pairs (Karollus et al. 2023 windowing via ziga
tables).

Base-model evaluation only. See README for adding new models.

Cerberus is Borzoi's successor: Borzoi's transformer tower and U-net decoder are
replaced by 8 Hydra blocks (bidirectional Mamba-2 state space mixers) run at
32 bp resolution.

  - Input: 786,432 bp. Output: 8,361 human tracks at 32 bp over the *whole*
    input (24,576 bins, no crop), so the TSS bin is ``tss_seq_index // 32``.
  - Released as an ensemble of 8 replicate models (``f0c0`` .. ``f7c0``), each
    holding out a different fold. Every pair is scored with all 8 replicates;
    ``pred_wt`` and ``pred_crispri_mean`` are averaged across replicates before
    ``pred_delta`` is computed (the ``make_borzoi_ensemble.py`` recipe).
  - Readout (``--readout``), as in the Borzoi scripts, forward strand only (no
    reverse-complement averaging):
      ``exon`` (default) -- coverage converted back from the model's squashed
        units with Calico's ``untransform_preds`` (``(y + 1)^2 - 1`` for these
        tracks, ``unscale=False``), summed over the output bins overlapping the
        gene's GENCODE exons (``--exon_csv``), strand-matched tracks only ('+'
        and unstranded; minus-strand genes are reverse-complemented, so every
        gene is on '+'). The AlphaGenome / NTv3 RNA-Seq readout and Calico's own
        gene score. No exon bin in the window -> TSS window (``readout`` column).
      ``tss`` -- all tracks of the set, transformed units, summed over a
        TSS-centred window of 32 bp bins. Default prefix gains ``_tss``.
  - Two track sets are scored from the same forward passes:
      ``rna``          -- the 94 K562 RNA-Seq tracks Borzoi uses, matched by
                          ENCODE identifier against ``--borzoi_target_csv``.
      ``rna_allk562``  -- all 160 K562 RNA tracks; adds 66 recount3 tracks
                          (4sU / TT-seq nascent RNA, actinomycin D shutoff).
  - Per-track sums for every replicate (all 160 tracks, before the strand
    filter) are saved to ``<prefix>_per_track.npz`` so other track subsets can
    be re-aggregated offline without re-running the model.
  - Runs in fp32 unless ``--autocast_dtype`` is given. PyTorch TF32 is turned
    off, but mamba-ssm's Triton kernels still use TF32 (set
    ``TRITON_F32_DEFAULT=ieee`` for strict fp32: ~30% slower, and it moves
    per-replicate pred_delta by <= 2.4e-4 here).

Requires calico/baskerville installed with the ``cuda`` extra (mamba-ssm) and
the released weights:

    cd baskerville/releases/cerberus && ./download.sh

``--cerberus_dir`` points at that directory (``params.json`` +
``models/f<k>c0/model_best.pth``).

Usage:
    python scripts/test_fulco_cerberus.py \\
        --cerberus_dir /path/to/baskerville/releases/cerberus

    # Quick check on one gene with two replicates
    python scripts/test_fulco_cerberus.py --cerberus_dir ... --gene PQBP1 --folds 0 1

    # One of 2 shards (fp32 is ~3 min per pair on an H100); merge each track set
    # with scripts/merge_tta_shards.py --fulco_corr_observed_subset knockdown_only
    python scripts/test_fulco_cerberus.py --cerberus_dir ... --num_shards 2 --shard_idx 0
"""

import argparse
import contextlib
import io
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from crispri_eval import readouts
from crispri_eval.datasets import FulcoDataset
from crispri_eval.benchmark_utils import fulco_results_for_correlation


# Cerberus (calico/baskerville releases/cerberus) constants
CERBERUS_SEQ_LEN = 786_432            # input bp
CERBERUS_BIN_BP = 32                  # output bin width (bp)
CERBERUS_NUM_BINS = CERBERUS_SEQ_LEN // CERBERUS_BIN_BP  # 24,576; no output crop
CERBERUS_NUM_FOLDS = 8

# Same TSS window as the Borzoi scripts: 20 // 2 = 10 bins either side of the
# TSS bin (21 x 32 bp).
DEFAULT_CENTRAL_BINS = 20

TRACK_SETS = ("rna", "rna_allk562")


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def _tss_bin_from_seq_index(tss_seq_index, num_bins=CERBERUS_NUM_BINS):
    """Map an input-sequence index (0..786431) to an output bin index (0..24575)."""
    bin_idx = int(tss_seq_index) // CERBERUS_BIN_BP
    if bin_idx < 0 or bin_idx >= num_bins:
        warnings.warn(
            "TSS at sequence index {} falls outside the input window; clamping "
            "the readout bin to the nearest edge.".format(tss_seq_index)
        )
    return int(max(0, min(num_bins - 1, bin_idx)))


def parse_args():
    p = argparse.ArgumentParser(
        description="Test Cerberus on Fulco CRISPRi benchmark (Karollus windows)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model ----------------------------------------------------------------
    p.add_argument(
        "--cerberus_dir", type=str,
        default=os.environ.get("CERBERUS_DIR", "./models/cerberus"),
        help=(
            "baskerville releases/cerberus directory holding params.json and "
            "models/f<k>c0/model_best.pth (from download.sh). "
            "Defaults to $CERBERUS_DIR if set."
        ),
    )
    p.add_argument(
        "--folds", type=int, nargs="+", default=list(range(CERBERUS_NUM_FOLDS)),
        help="Replicate folds to ensemble (default: all 8).",
    )

    # -- Track selection ------------------------------------------------------
    p.add_argument(
        "--target_csv", type=str, default="./metadata/cerberus_human_targets.txt",
        help="Cerberus targets_human.txt (row order = head output order).",
    )
    p.add_argument(
        "--borzoi_target_csv", type=str, default="./metadata/borzoi_human_targets.txt",
        help="Borzoi targets_human.txt; defines the Borzoi-matched 'rna' track set.",
    )
    p.add_argument(
        "--target_cell_line", type=str, default="K562",
        help="Cell line to match in the description column of the targets tables.",
    )
    p.add_argument(
        "--num_central_bins", type=int, default=DEFAULT_CENTRAL_BINS,
        help=(
            "Number of 32 bp bins to aggregate around the TSS bin. Default {} "
            "(same window as the Borzoi scripts).".format(DEFAULT_CENTRAL_BINS)
        ),
    )
    p.add_argument(
        "--readout", type=str, default="exon", choices=["exon", "tss"],
        help=(
            "Gene readout. 'exon': strand-matched tracks, untransformed, summed over the "
            "gene's exon bins. 'tss': all tracks, transformed units, summed over "
            "--num_central_bins around the TSS."
        ),
    )
    p.add_argument(
        "--exon_csv", type=str, default="./metadata/Fulco_gene_exons_hg38.csv",
        help="Exon table [gene_id, chrom, start, end] (0-based half-open) for --readout exon.",
    )

    p.add_argument(
        "--shuffle_batch_size", type=int, default=2,
        help=(
            "Number of dinucleotide-shuffled CRISPRi sequences per forward pass. "
            "Each 786,432 bp sequence produces a 8361 x 24576 coverage map."
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
    p.add_argument("--seed", type=int, default=101,
        help="Seed for the enhancer shuffles, drawn per pair so every model sees the same shuffled sequences (default: 101)")
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
    p.add_argument(
        "--num_shards", type=int, default=1,
        help="Split the enhancer-gene pairs into this many strided shards (for cluster arrays).",
    )
    p.add_argument(
        "--shard_idx", type=int, default=0,
        help="0-based shard index in [0, num_shards). Each shard processes pairs[shard_idx::num_shards].",
    )

    # -- Output ---------------------------------------------------------------
    p.add_argument(
        "--save_path", type=str, default="./results/test_Fulco_cerberus/",
        help="Output directory.",
    )
    p.add_argument(
        "--output_prefix", type=str, default=None,
        help="Base prefix; the track-set name is appended (default: Fulco_Cerberus).",
    )
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
            "Run forward passes under torch.autocast(dtype=...). Default: fp32. "
            "Eager bf16 corrupts small WT-vs-shuffle deltas on other models."
        ),
    )

    args = p.parse_args()
    if not all(0 <= f < CERBERUS_NUM_FOLDS for f in args.folds):
        p.error("--folds must be in [0, {})".format(CERBERUS_NUM_FOLDS))
    if args.num_shards < 1:
        p.error("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        p.error("--shard_idx must be in [0, num_shards)")
    return args


# ---------------------------------------------------------------------------
# Cerberus targets_human.txt -> K562 RNA track sets
# ---------------------------------------------------------------------------

def get_cerberus_track_sets(target_csv, borzoi_target_csv, cell_line):
    """Return (superset head indices, {set name: positions within superset}, identifiers).

    The superset is every Cerberus 'rna' track whose description mentions
    *cell_line*. ``rna`` keeps the ones whose identifier is a Borzoi
    ``RNA:<cell_line>`` track; ``rna_allk562`` keeps them all.
    """
    for path in (target_csv, borzoi_target_csv):
        if not os.path.exists(path):
            raise FileNotFoundError("Targets file not found: {}".format(path))

    df = pd.read_csv(target_csv, sep="\t", index_col=0)
    if not np.array_equal(df.index.values, np.arange(len(df))):
        raise ValueError("{}: first column must be the head output index 0..N-1".format(target_csv))
    desc = df["description"].astype(str)
    superset = df[(df["assay"] == "rna") & desc.str.contains(cell_line, case=False, na=False)]

    bz = pd.read_csv(borzoi_target_csv, sep="\t", index_col=0)
    bz_desc = bz["description"].astype(str)
    bz_ids = set(bz.loc[
        bz_desc.str.startswith("RNA:") & bz_desc.str.contains(cell_line, case=False, na=False),
        "identifier",
    ])
    in_borzoi = superset["identifier"].isin(bz_ids).values
    if in_borzoi.sum() != len(bz_ids):
        raise ValueError(
            "Only {} of {} Borzoi {} RNA tracks found in {}.".format(
                in_borzoi.sum(), len(bz_ids), cell_line, target_csv
            )
        )

    track_sets = {
        "rna": np.flatnonzero(in_borzoi),
        "rna_allk562": np.arange(len(superset)),
    }
    return superset.index.values.astype(int), track_sets, superset["identifier"].tolist()


def describe_track_set(name, n_tracks, cell_line):
    if name == "rna":
        return "{} {} RNA-Seq tracks (Borzoi-matched ENCODE identifiers)".format(n_tracks, cell_line)
    return "{} {} RNA tracks (all; Borzoi's ENCODE set + recount3 4sU/TT-seq/ActD)".format(
        n_tracks, cell_line
    )


# ---------------------------------------------------------------------------
# Model loading / inference
# ---------------------------------------------------------------------------

def load_cerberus_models(cerberus_dir, folds, device):
    """Load the requested Cerberus replicates (strict state-dict match)."""
    from baskerville.seqnn import SeqNN

    params_json = os.path.join(cerberus_dir, "params.json")
    with open(params_json) as f:
        params = json.load(f)
    heads = sorted(k for k in params["model"] if k.startswith("head"))
    head_index = heads.index("head_human")

    models = []
    for fold in folds:
        weights = os.path.join(cerberus_dir, "models", "f{}c0".format(fold), "model_best.pth")
        print("Loading Cerberus replicate f{}c0: {}".format(fold, weights))
        # SeqNN.__init__ prints the full module.
        with contextlib.redirect_stdout(io.StringIO()):
            snn = SeqNN(params["model"])
        snn.set_device(device)
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        snn.model.load_state_dict(sd, strict=True)
        snn.model.eval().to(device)
        models.append(snn.model)
    return models, head_index


def _autocast_ctx(device, autocast_dtype):
    if autocast_dtype is None or device == "cpu":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if autocast_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _predict_window_sums(model, head_index, x, track_inds, bins, device, autocast_dtype,
                         untransform=None):
    """(B, L, 4) one-hot -> (B, n_tracks) coverage summed over output ``bins`` (LongTensor).

    ``untransform`` maps the selected tracks back to coverage units before summing.
    """
    x_dev = x.permute(0, 2, 1).contiguous().to(device)
    with torch.no_grad(), _autocast_ctx(device, autocast_dtype):
        out = model(x_dev, head_index)
    cov = out.coverage[:, track_inds, :].float()
    if untransform is not None:
        cov = untransform(cov)
    return cov[:, :, bins].sum(dim=2).cpu()


def score_pair(models, head_index, x_wt, x_crispri, track_inds, bins, bs, device, autocast_dtype,
               untransform=None):
    """Per-replicate, per-track sums over ``bins``: (wt, crispri_mean, crispri_sums).

    wt, crispri_mean: (n_folds, n_tracks); crispri_sums: (n_folds, N, n_tracks).
    """
    wt, crispri = [], []
    for model in models:
        wt.append(_predict_window_sums(
            model, head_index, x_wt, track_inds, bins, device, autocast_dtype, untransform)[0])
        crispri.append(torch.cat([
            _predict_window_sums(
                model, head_index, x_crispri[i : i + bs], track_inds, bins,
                device, autocast_dtype, untransform)
            for i in range(0, x_crispri.shape[0], bs)
        ]))
    crispri = torch.stack(crispri).numpy()
    return torch.stack(wt).numpy(), crispri.mean(axis=1), crispri


def _tss_window_bins(tss_bin, central_bins, n_bins=CERBERUS_NUM_BINS):
    """Output bins of the TSS window: ``central_bins // 2`` either side of ``tss_bin``."""
    bin_offset = central_bins // 2
    return np.arange(max(0, tss_bin - bin_offset), min(n_bins, tss_bin + bin_offset + 1))


def setup_readout(args, track_inds, device):
    """(strand-matched mask over the superset, untransform, exon table) for ``--readout``.

    The ``tss`` readout keeps every track and the transformed units.
    """
    if args.readout == "tss":
        return np.ones(len(track_inds), dtype=bool), None, None
    sm, untransform = readouts.cerberus_exon_setup(args.target_csv, track_inds, device)
    return sm, untransform, readouts.load_exon_table(args.exon_csv)


def aggregation_description(args):
    if args.readout == "exon":
        return ("exon: untransformed coverage of strand-matched tracks summed over the output "
                "bins overlapping the gene's exons ({}); TSS bin +/- {} bins if none".format(
                    args.exon_csv, args.num_central_bins // 2))
    return "tss: transformed units summed over TSS bin +/- {} bins ({} bp bins)".format(
        args.num_central_bins // 2, CERBERUS_BIN_BP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    if args.autocast_dtype is None:
        # fp32 means fp32: TF32 matmuls/convs keep only 10 mantissa bits.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    os.makedirs(args.save_path, exist_ok=True)

    # ---- Model loading -------------------------------------------------------
    models, head_index = load_cerberus_models(args.cerberus_dir, args.folds, device)
    print("Loaded {} Cerberus replicate(s): folds {}".format(len(models), args.folds))

    # ---- Track selection -----------------------------------------------------
    track_inds, track_sets, track_ids = get_cerberus_track_sets(
        args.target_csv, args.borzoi_target_csv, args.target_cell_line
    )
    sm_mask, untransform, exon_coords = setup_readout(args, track_inds, device)
    for name in TRACK_SETS:
        print("Track set '{}': {}{}".format(
            name, describe_track_set(name, len(track_sets[name]), args.target_cell_line),
            "" if args.readout == "tss" else
            " | {} strand-matched used".format(int(sm_mask[track_sets[name]].sum()))))
    track_inds_tensor = torch.as_tensor(track_inds, dtype=torch.long, device=device)
    print("Aggregation: {}".format(aggregation_description(args)))

    # ---- Dataset -------------------------------------------------------------
    print("\nLoading Fulco dataset from {}...".format(args.data_path))
    print("Sequence window: centred on ziga TSS (Karollus et al.); enhancer shuffled in place.")
    dataset = FulcoDataset(
        data_path=args.data_path,
        sequence_length=CERBERUS_SEQ_LEN,
        N=args.N,
        seed=args.seed,
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

    # ---- Shard the pairs (for SLURM array parallelisation) ------------------
    if args.num_shards > 1:
        n_all = len(dataset.crispri_data)
        dataset.crispri_data = (
            dataset.crispri_data.iloc[args.shard_idx::args.num_shards].reset_index(drop=True)
        )
        print("Shard {}/{}: {} of {} pairs".format(
            args.shard_idx, args.num_shards, len(dataset.crispri_data), n_all
        ))

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,  # pre-batched
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # ---- Inference -----------------------------------------------------------
    print("\nRunning inference on {} enhancer-gene pairs...".format(len(dataset)))
    print("  {} dinucleotide shuffles per pair x {} replicates".format(args.N, len(models)))
    print("  Context: {:,} bp  |  shuffle batch size: {}  |  autocast: {}".format(
        CERBERUS_SEQ_LEN, args.shuffle_batch_size, args.autocast_dtype or "off (fp32)"
    ))

    meta = {k: [] for k in (
        "gene_id", "gene_name", "enh_loc", "tss", "tss_seq_index", "tss_bin",
        "readout", "n_readout_bins", "strand", "enh_dist", "y_delta",
    )}
    all_wt, all_crispri_mean, all_crispri_std = [], [], []
    bs = max(1, int(args.shuffle_batch_size))

    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue

        tss_seq_idx = _tss_seq_index_from_batch(batch)
        tss_bin = _tss_bin_from_seq_index(tss_seq_idx)
        bins, readout_used = readouts.pair_readout_bins(
            args.readout, batch, exon_coords, _tss_window_bins(tss_bin, args.num_central_bins),
            CERBERUS_SEQ_LEN, 0, CERBERUS_BIN_BP, CERBERUS_NUM_BINS,
        )
        bins_t = torch.as_tensor(bins, dtype=torch.long, device=device)

        wt, crispri_mean, crispri = score_pair(
            models, head_index, batch["x_WT"], batch["x_crispri"], track_inds_tensor,
            bins_t, bs, device, args.autocast_dtype, untransform,
        )
        all_wt.append(wt)
        all_crispri_mean.append(crispri_mean)
        # Spread of the ensemble-mean, all-track shuffle predictions (debug only).
        all_crispri_std.append(float(crispri.sum(axis=2).mean(axis=0).std()))

        meta["gene_id"].append(batch["gene_id"])
        meta["gene_name"].append(batch["gene_name"])
        meta["enh_loc"].append(batch["enh_loc"])
        meta["tss"].append(batch["tss"])
        meta["tss_seq_index"].append(tss_seq_idx)
        meta["tss_bin"].append(tss_bin)
        meta["readout"].append(readout_used)
        meta["n_readout_bins"].append(len(bins))
        meta["strand"].append(batch["strand"])
        meta["enh_dist"].append(batch["enh_dist"])
        meta["y_delta"].append((1.0 - batch["y_delta"]).item())

    if len(all_wt) == 0:
        print("No pairs evaluated -- exiting.")
        return None

    return write_outputs(
        args, meta, np.stack(all_wt), np.stack(all_crispri_mean), all_crispri_std,
        track_inds, track_sets, track_ids, sm_mask,
    )


def write_outputs(args, meta, all_wt, all_crispri_mean, all_crispri_std,
                  track_inds, track_sets, track_ids, sm_mask):
    """Write the per-track npz and one results CSV (+ summary) per track set.

    all_wt, all_crispri_mean: (pairs, folds, tracks) per-track sums over the readout
    bins, tracks in ``track_inds`` order. ``sm_mask`` (tracks,) selects the tracks
    summed into each set (all True for ``--readout tss``).
    """
    base_prefix = args.output_prefix or (
        "Fulco_Cerberus" + ("_tss" if args.readout == "tss" else ""))
    shard_tag = (
        "_shard{:03d}of{:03d}".format(args.shard_idx, args.num_shards)
        if args.num_shards > 1 else ""
    )
    npz_path = os.path.join(args.save_path, "{}{}_per_track.npz".format(base_prefix, shard_tag))
    np.savez_compressed(
        npz_path,
        pred_wt=all_wt,
        pred_crispri_mean=all_crispri_mean,
        folds=np.asarray(args.folds),
        track_head_index=track_inds,
        track_identifier=np.asarray(track_ids),
        track_strand_matched=np.asarray(sm_mask),
        readout_mode=np.asarray(args.readout),
        readout=np.asarray(meta["readout"]),
        n_readout_bins=np.asarray(meta["n_readout_bins"]),
        gene_id=np.asarray(meta["gene_id"]),
        enh_loc=np.asarray(meta["enh_loc"]),
        **{"set_" + name: idx for name, idx in track_sets.items()},
    )
    print("\nPer-track sums saved to: {}".format(npz_path))

    n_fallback = sum(r != args.readout for r in meta["readout"])
    if n_fallback:
        print("{} / {} pairs had no exon bin in the window and used the TSS window.".format(
            n_fallback, len(meta["readout"])))

    # ---- Results (one table per track set) ------------------------------------
    print("\nCompiling results...")
    out = {}
    for name in TRACK_SETS:
        idx = track_sets[name]
        idx = idx[sm_mask[idx]]
        # Sum the set's tracks, then average replicates (make_borzoi_ensemble.py).
        pred_wt = all_wt[:, :, idx].sum(axis=2).mean(axis=1)
        pred_crispri_mean = all_crispri_mean[:, :, idx].sum(axis=2).mean(axis=1)
        df_results = pd.DataFrame(meta)
        df_results["pred_delta"] = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)
        df_results["pred_wt"] = pred_wt
        df_results["pred_crispri_mean"] = pred_crispri_mean
        if args.debug_export:
            df_results["debug_track_count_used"] = len(idx)
            df_results["debug_crispri_std_alltracks"] = all_crispri_std

        output_prefix = "{}_{}".format(base_prefix, name)
        # In sharded mode, write only this shard's per-pair CSV; the merge step
        # (scripts/merge_tta_shards.py --fulco_corr_observed_subset ...) concatenates
        # shards and computes correlations.
        if args.num_shards > 1:
            shard_csv = os.path.join(
                args.save_path, "{}{}_results.csv".format(output_prefix, shard_tag)
            )
            df_results.to_csv(shard_csv, index=False)
            print("Shard {}/{} '{}' results saved to: {}".format(
                args.shard_idx, args.num_shards, name, shard_csv))
            out[name] = df_results
            continue
        out[name] = _report_and_save(df_results, output_prefix, name, len(track_sets[name]),
                                     len(idx), n_fallback, args)
    return out


def _report_and_save(df_results, output_prefix, set_name, n_tracks, n_used, n_fallback, args):
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

    track_desc = describe_track_set(set_name, n_tracks, args.target_cell_line)
    if n_used != n_tracks:
        track_desc += "; {} strand-matched used".format(n_used)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY -- track set '{}': {}".format(set_name, track_desc))
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

    csv_path = os.path.join(args.save_path, "{}_results.csv".format(output_prefix))
    df_results.to_csv(csv_path, index=False)
    print("\nResults saved to: {}".format(csv_path))

    summary_path = os.path.join(args.save_path, "{}_summary.txt".format(output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULCO BENCHMARK SUMMARY -- Cerberus\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write("Model:             Cerberus (calico/baskerville)\n")
        f.write("Weights:           {}\n".format(args.cerberus_dir))
        f.write("Replicates:        {} (folds {}), pred_wt / pred_crispri averaged\n".format(
            len(args.folds), args.folds))
        f.write("Autocast:          {}\n".format(
            args.autocast_dtype or "off (fp32; mamba-ssm Triton kernels use TF32)"))
        f.write("\nEVALUATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("Context window:    {:,} bp\n".format(CERBERUS_SEQ_LEN))
        f.write("Bin resolution:    {} bp ({} output bins, no crop)\n".format(
            CERBERUS_BIN_BP, CERBERUS_NUM_BINS))
        f.write("Genome build:      {}\n".format(args.genome_build))
        f.write("Fulco data path:   {}\n".format(args.data_path))
        f.write("Gene TSS CSV:      {}\n".format(args.gene_tss_csv))
        f.write("Validated only:    {}\n".format(args.validated_only))
        f.write("N shuffles:        {}\n".format(args.N))
        f.write("Shuffle seed:      {}\n".format(args.seed))
        f.write("Enhancer window:   {} bp\n".format(args.enhancer_bp))
        f.write("Target cell line:  {}\n".format(args.target_cell_line))
        f.write("Track set:         {}: {}\n".format(set_name, track_desc))
        f.write("Aggregation:       {}\n".format(aggregation_description(args)))
        if args.readout == "exon":
            f.write("TSS-window fallback: {} / {} pairs\n".format(n_fallback, n_full))
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
