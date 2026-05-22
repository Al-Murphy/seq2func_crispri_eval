#!/usr/bin/env python3
"""
Test base AlphaGenome on Fulco et al. (2019) CRISPRi enhancer knockdown pairs
(Karollus et al. 2023 windowing via ziga tables).

Base-model evaluation only. See README for adding new models.

AlphaGenome includes RNA-Seq tracks for K562 (erythroleukemia cell line), making
it a strong candidate for this CRISPRi perturbation benchmark.

  - RNA-Seq predictions at 1 bp resolution (full U-Net decoder) are used.
  - K562 tracks are identified via a pre-extracted track-metadata parquet.
  - Signal is summed over a TSS-centred window and selected K562 tracks. Use
    ``--modality rna_seq`` (default) or ``--modality cage`` for CAGE-seq; CAGE
    uses a fixed 5×128 bp window matching ``test_fulco_enformer.py`` defaults.
  - Experimental labels match Enformer: y_delta = 1 - expr_ratio; expr_ratio = 1 + Fulco fraction change
    (higher = stronger repression vs WT). Pearson/Spearman rows follow
    ``--fulco_corr_observed_subset`` (default ``knockdown_only``).
  - Predicted delta matches Enformer/Borzoi: fractional drop in summed signal,
    pred_delta = (pred_WT - pred_CRISPRi_mean) / (pred_WT + 1e-8).

  Note (AlphaGenome paper, variant scoring): RNA-Seq reads are typically
  summarised in **log fold change**. The script still saves linear ``pred_delta``
  for parity with ``test_fulco_enformer.py``, but predicted fractional changes
  from exon means are often ~1e-4 while experimental ``y_delta`` is ~0.01–1 —
  **do not rely on linear Pearson vs y_delta alone**. Prefer the printed
  **log2 knockdown alignment** (matches Karollus observed ``log2(expr_ratio)``
  direction). CAGE modality uses the same linear score with a fixed TSS-sum
  window closer to Enformer scales.

  Sequence window: centred on the TSS from ``ziga_additional_columns.tsv``.
  Strand for reverse-complement is from the Biomart ``--gene_tss_csv`` table.
  The enhancer is shuffled in place within this TSS-centred window.

Usage:
    # Minimal invocation; backbone and metadata are auto-fetched.
    python scripts/test_fulco_alphagenome.py \\
        --data_path ./metadata/fulco/ \\
        --gene_tss_csv ./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv

    # Explicit local weights; metadata still auto-generated if absent
    python scripts/test_fulco_alphagenome.py \\
        --backbone_model_path /path/to/model_all_folds.safetensors

    # Custom metadata location
    python scripts/test_fulco_alphagenome.py \\
        --metadata_path /path/to/track_metadata.parquet

    # CAGE (K562), 5×128 bp TSS window — comparable to Enformer Fulco benchmark
    python scripts/test_fulco_alphagenome.py --modality cage
"""

import argparse
import json
import os
import urllib.request as _urlreq
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from crispri_eval.datasets import FulcoDataset
from crispri_eval.benchmark_utils import fulco_results_for_correlation


# AlphaGenome's full context window (2^20 = 1,048,576 bp)
AG_SEQ_LEN = 1_048_576

# Match test_fulco_enformer.py: --num_central_bins 5, 128 bp bins → 640 bp at TSS
ENFORMER_CENTRAL_BINS = 5
ENFORMER_BIN_BP = 128
ENFORMER_TSS_WINDOW_BP = ENFORMER_CENTRAL_BINS * ENFORMER_BIN_BP  # 640

# Human AlphaGenome 1 bp head widths (must match pretrained checkpoints)
AG_HUMAN_1BP_NUM_TRACKS = {"rna_seq": 768, "cage": 640}


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def parse_args():
    p = argparse.ArgumentParser(
        description="Test base AlphaGenome on Fulco CRISPRi benchmark (Karollus windows)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--backbone_model_path", type=str, default=None,
        help=(
            "Path to AlphaGenome pretrained weights (.safetensors or .pth). "
            "If omitted, the all-folds model is downloaded from HuggingFace "
            "(``gtca/alphagenome_pytorch / model_all_folds.safetensors``)."
        ),
    )
    p.add_argument("--organism_index", type=int, default=0,
                   help="0 = human (default), 1 = mouse.")

    p.add_argument(
        "--modality", choices=["rna_seq", "cage"], default="rna_seq",
        help=(
            "Which 1 bp AlphaGenome head to read: rna_seq (768 tracks) or cage (640 tracks). "
            "cage uses K562 CAGE tracks and a fixed TSS window of {} bp ({} × {} bp); "
            "rna_seq uses --window_bp (default {} bp = same Enformer-equivalent width)."
        ).format(
            ENFORMER_TSS_WINDOW_BP, ENFORMER_CENTRAL_BINS, ENFORMER_BIN_BP, ENFORMER_TSS_WINDOW_BP,
        ),
    )
    p.add_argument(
        "--metadata_path", type=str,
        default="./metadata/track_metadata.parquet",
        help=(
            "Path to track_metadata.parquet (produced by "
            "alphagenome-pytorch/scripts/extract_track_metadata.py). "
            "Used for K562 track selection. Set to '' or 'none' to skip cell-line "
            "filtering and use all tracks for the selected --modality."
        ),
    )
    p.add_argument(
        "--target_cell_line", type=str, default="K562",
        help="Cell line to match in biosample_name of the track metadata.",
    )
    p.add_argument(
        "--window_bp", type=int, default=ENFORMER_TSS_WINDOW_BP,
        help=(
            "TSS-centred window in bp (1 bp resolution outputs). Only used when "
            "--modality rna_seq and no exon coords are available. "
            "Default {} bp matches Enformer ({} central bins × {} bp). "
            "Scoring: pred_delta = (pred_WT - pred_CRISPRi_mean) / (pred_WT + 1e-8)."
        ).format(ENFORMER_TSS_WINDOW_BP, ENFORMER_CENTRAL_BINS, ENFORMER_BIN_BP),
    )
    p.add_argument(
        "--shuffle_batch_size", type=int, default=2,
        help=(
            "Number of dinucleotide-shuffled CRISPRi sequences to run through the "
            "model in a single forward pass. Increases throughput at the cost of GPU "
            "memory (~13 GB per extra sequence at 1 bp resolution / 1 Mb input)."
        ),
    )

    p.add_argument(
        "--data_path", type=str, default="./metadata/fulco/",
        help="Directory with ziga_additional_columns.tsv and enhancer_knockdown_effects.tsv",
    )
    p.add_argument(
        "--gene_tss_csv", type=str,
        default="./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv",
        help="Biomart CSV with Gene stable ID and Strand (1 / -1).",
    )
    p.add_argument("--genome_build", type=str, default="hg38", choices=["hg38", "hg19"])
    p.add_argument("--N", type=int, default=50,
                   help="Number of dinucleotide shuffles per enhancer.")
    p.add_argument("--enhancer_bp", type=int, default=2000,
                   help="Size of enhancer region to shuffle (bp).")
    p.add_argument(
        "--crispri_perturb_mode", type=str, default="dinucleotide",
        choices=["dinucleotide", "shuffle_bases", "random_permute"],
        help="CRISPRi negatives: dinucleotide (default); shuffle_bases / random_permute.",
    )
    p.add_argument(
        "--validated_only", type=bool, default=True,
        help="If True, keep only validated==True rows in ziga (default: True).",
    )
    p.add_argument(
        "--min_enh_dist", type=int, default=990, metavar="BP",
        help=(
            "Minimum enhancer–TSS distance (bp, exclusive). Pairs with enh_dist <= this "
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
        "--exon_csv", type=str,
        default="./metadata/Fulco_gene_exons_hg38.csv",
        help=(
            "CSV with columns [gene_id, chrom, start, end] (0-based half-open) for "
            "GENCODE exon coordinates. Used for RNA-Seq aggregation (AlphaGenome paper: "
            "mean signal across exons). If absent, auto-generated via Ensembl REST API. "
            "Pass 'none' to fall back to --window_bp TSS-window scoring."
        ),
    )

    p.add_argument("--save_path", type=str,
                   default="./results/test_Fulco_alphagenome/")
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument(
        "--debug_export", action="store_true",
        help="Include per-pair diagnostics (exon bp, track counts, WT/shuffle spread).",
    )

    # -- Test-time augmentation (TTA) -----------------------------------------
    p.add_argument(
        "--tta_shifts", type=str, default="0",
        help=(
            "Comma-separated bp shifts for test-time augmentation. Each shift "
            "re-fetches a real-flank window at seq_start+shift (no rolling/wrapping). "
            "Default '0' = no shift. Karollus 2023 used '-43,0,43' (1/3-bin sampling for Enformer)."
        ),
    )
    p.add_argument(
        "--tta_rev_comp", action="store_true",
        help=(
            "Add reverse-complement passes to TTA (flip the sequence, read minus-strand "
            "RNA-Seq tracks at flipped positions). With --tta_shifts -43,0,43 this gives "
            "the 6-pass Karollus-matched augmentation."
        ),
    )

    # -- Sharding (for SLURM array parallelisation) ---------------------------
    p.add_argument(
        "--num_shards", type=int, default=1,
        help="Split the enhancer-gene pairs into this many strided shards (for cluster arrays).",
    )
    p.add_argument(
        "--shard_idx", type=int, default=0,
        help="0-based shard index in [0, num_shards). Each shard processes pairs[shard_idx::num_shards].",
    )

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)

    args = p.parse_args()

    if args.num_shards < 1:
        p.error("--num_shards must be >= 1")
    if not (0 <= args.shard_idx < args.num_shards):
        p.error("--shard_idx must be in [0, num_shards)")

    return args


# ---------------------------------------------------------------------------
# Track metadata -> K562 track indices
# ---------------------------------------------------------------------------

def get_alphagenome_track_indices(metadata_path, cell_line, output_type, n_total_tracks):
    """
    Return (track index bundle, description) for tracks matching *cell_line*.

    The track_metadata.parquet is produced by
    ``alphagenome-pytorch/scripts/extract_track_metadata.py``. Row order for a
    given (organism, output_type) block matches the model head index
    (e.g. rna_seq: B x S x 768, cage: B x S x 640).

    Parameters
    ----------
    output_type : str
        Lowercase column value, e.g. ``"rna_seq"`` or ``"cage"``.
    n_total_tracks : int
        Expected number of human tracks for that modality (768 or 640).
    """
    assay_label = "RNA-Seq" if output_type.lower() == "rna_seq" else output_type.upper()

    if metadata_path is None or str(metadata_path).strip().lower() in ("", "none"):
        warnings.warn(
            "No metadata path given. Using all {} human {} tracks.".format(
                n_total_tracks, assay_label
            )
        )
        all_inds = list(range(n_total_tracks))
        return (
            {"all": all_inds, "plus": all_inds, "minus": all_inds},
            "all {} human {} tracks".format(n_total_tracks, assay_label),
        )

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            "Track metadata not found: {}\n\n"
            "Generate it using the alphagenome_research (JAX) environment:\n"
            "  cd /path/to/alphagenome-pytorch\n"
            "  python scripts/extract_track_metadata.py --output-file {}\n\n"
            "Or place an existing track_metadata.parquet at "
            "./metadata/track_metadata.parquet (the default --metadata_path).".format(
                metadata_path, metadata_path
            )
        )

    df = pd.read_parquet(metadata_path)

    mask_human = df["organism"].str.lower() == "human"
    mask_type = df["output_type"].str.lower() == output_type.lower()
    df_m = df[mask_human & mask_type].reset_index(drop=True)

    if len(df_m) == 0:
        raise ValueError(
            "No human {} tracks found in the metadata (output_type='{}').".format(
                assay_label, output_type
            )
        )

    if len(df_m) != n_total_tracks:
        warnings.warn(
            "Expected {} human {} tracks but found {} in metadata.".format(
                n_total_tracks, assay_label, len(df_m)
            )
        )

    search_col = "biosample_name" if "biosample_name" in df_m.columns else "track_name"
    mask_cl = df_m[search_col].str.contains(cell_line, case=False, na=False)
    df_cl = df_m[mask_cl]

    if len(df_cl) == 0:
        warnings.warn(
            "No {} tracks matched '{}' in column '{}'. Using all {} human {} tracks.".format(
                assay_label, cell_line, search_col, len(df_m), assay_label
            )
        )
        all_inds = df_m.index.tolist()
        return (
            {"all": all_inds, "plus": all_inds, "minus": all_inds},
            "all {} human {} tracks (no {} match)".format(
                len(df_m), assay_label, cell_line
            ),
        )

    indices = df_cl.index.tolist()
    track_names = df_cl[search_col].tolist()
    extra = " (+{} more)".format(len(indices) - 4) if len(indices) > 4 else ""
    desc = "{} {} {} tracks: {}{}".format(
        len(indices), cell_line, assay_label, ", ".join(track_names[:4]), extra
    )

    if "track_strand" not in df_cl.columns:
        return (
            {"all": indices, "plus": indices, "minus": indices},
            desc + " (no track_strand metadata; strand-unaware)",
        )

    track_strand = df_cl["track_strand"].astype(str).fillna(".")
    plus_indices = df_cl[track_strand.isin(["+", "."])].index.tolist()
    minus_indices = df_cl[track_strand.isin(["-", "."])].index.tolist()
    if len(plus_indices) == 0:
        plus_indices = indices
    if len(minus_indices) == 0:
        minus_indices = indices

    return (
        {"all": indices, "plus": plus_indices, "minus": minus_indices},
        "{} | strand-aware (+/.: {}, -/.: {})".format(desc, len(plus_indices), len(minus_indices)),
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_model(backbone_path, device):
    """Load pretrained AlphaGenome.

    If *backbone_path* is None, downloads the all-folds weights from HuggingFace
    (``gtca/alphagenome_pytorch`` / ``model_all_folds.safetensors``). Always go
    through ``from_pretrained`` — calling ``alphagenome_pytorch.AlphaGenome()``
    bare would silently return a randomly-initialised model.
    """
    from alphagenome_pytorch import AlphaGenome
    if backbone_path is None:
        from huggingface_hub import hf_hub_download
        print("Loading base AlphaGenome (HuggingFace all-folds weights via gtca/alphagenome_pytorch)")
        backbone_path = hf_hub_download(
            repo_id="gtca/alphagenome_pytorch",
            filename="model_all_folds.safetensors",
        )
    else:
        print("Loading base AlphaGenome from: {}".format(backbone_path))
    model = AlphaGenome.from_pretrained(backbone_path, device=device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _forward_once(ag_model, x, organism_index, track_inds, device, modality):
    """One AlphaGenome forward pass; return (B, seq_len, n_selected) signal on the chosen tracks."""
    B = x.shape[0]
    org_idx = torch.full((B,), organism_index, dtype=torch.long, device=device)
    head_key = "cage" if modality == "cage" else "rna_seq"
    with torch.no_grad():
        outputs = ag_model(x.to(device), org_idx)
    if head_key not in outputs:
        raise KeyError(
            "AlphaGenome outputs missing {!r}; keys: {}".format(head_key, list(outputs.keys()))
        )
    return outputs[head_key][1][:, :, track_inds]  # (B, seq_len, n_selected) NLC


def _aggregate_sig(sig, modality, window_bp, center_index, exon_idx):
    """Reduce a (B, seq_len, n_tracks) signal tensor to one scalar per batch row.

    RNA-Seq with exon positions: mean over exon bp → mean over tracks (AlphaGenome paper).
    Otherwise (CAGE, or RNA-Seq with no exon mask): sum over a ``window_bp`` window
    centred on ``center_index``, then over tracks.
    """
    if modality == "rna_seq" and exon_idx:
        idx_t = torch.tensor(exon_idx, dtype=torch.long, device=sig.device)
        return sig[:, idx_t, :].mean(dim=1).mean(dim=1).tolist()

    L = sig.shape[1]
    center = L // 2 if center_index is None else max(0, min(L - 1, int(center_index)))
    half_lo = window_bp // 2
    half_hi = window_bp - half_lo
    start = max(0, center - half_lo)
    end = min(L, center + half_hi)
    return sig[:, start:end, :].sum(dim=(1, 2)).tolist()


# ---------------------------------------------------------------------------
# Exon annotation helpers (AlphaGenome RNA-Seq scoring approach)
# ---------------------------------------------------------------------------

def _fetch_exons_from_ensembl(gene_ids, genome_build="hg38"):
    """Fetch exon intervals from Ensembl REST API.

    Returns dict: {ensg_id: [(chrom_str, start_0based, end_excl), ...]}
    """
    server = (
        "https://rest.ensembl.org" if genome_build == "hg38"
        else "https://grch37.rest.ensembl.org"
    )
    result = {}
    for gid in gene_ids:
        url = "{}/overlap/id/{}?feature=exon;content-type=application/json".format(server, gid)
        try:
            req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with _urlreq.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            exons = list({
                (str(e.get("seq_region_name", "")), int(e["start"]) - 1, int(e["end"]))
                for e in data
                if "start" in e and "end" in e
            })
            result[gid] = exons
        except Exception as exc:
            warnings.warn("Could not fetch exons for {}: {}".format(gid, exc))
            result[gid] = []
    return result


def fetch_or_load_exon_coords(gene_ids, genome_build="hg38", cache_csv=None):
    """Return dict {ensg_id: [(chrom, start_0based, end_excl), ...]}.

    Loads from *cache_csv* when it exists (and fills any missing genes via the
    Ensembl REST API), otherwise fetches all genes and saves to *cache_csv*.
    """
    if cache_csv is None or str(cache_csv).strip().lower() == "none":
        return None

    if os.path.exists(cache_csv):
        print("Loading exon cache: {}".format(cache_csv))
        df = pd.read_csv(cache_csv)
        result = {}
        for gid, grp in df.groupby("gene_id"):
            result[gid] = list(zip(grp["chrom"], grp["start"].astype(int), grp["end"].astype(int)))
        missing = set(gene_ids) - set(result.keys())
        if missing:
            warnings.warn(
                "{} genes missing from exon cache; fetching from Ensembl: {}{}".format(
                    len(missing), list(missing)[:3], " ..." if len(missing) > 3 else ""
                )
            )
            extra = _fetch_exons_from_ensembl(missing, genome_build)
            result.update(extra)
            _save_exon_cache(result, cache_csv)
        return result

    print("Fetching exon annotations from Ensembl REST API ({} genes)...".format(len(gene_ids)))
    result = _fetch_exons_from_ensembl(list(gene_ids), genome_build)
    print("Saving exon cache to: {}".format(cache_csv))
    _save_exon_cache(result, cache_csv)
    return result


def _save_exon_cache(exon_coords, cache_csv):
    rows = []
    for gid, intervals in exon_coords.items():
        for chrom, start, end in intervals:
            rows.append({"gene_id": gid, "chrom": chrom, "start": start, "end": end})
    pd.DataFrame(rows).to_csv(cache_csv, index=False)


def _exon_seq_indices(gene_id, exon_coords, chrom, seq_start, seq_len, strand):
    """Map GENCODE exon intervals to 0-based positions in the one-hot tensor.

    For ``strand < 1`` the tensor was reverse-complemented, so positions are flipped.
    """
    gene_exons = exon_coords.get(gene_id, [])
    if not gene_exons:
        return []

    chrom_str = str(chrom).lstrip("chr")
    seq_end = seq_start + seq_len
    parts = []

    for ex_chrom, ex_start, ex_end in gene_exons:
        if str(ex_chrom).lstrip("chr") != chrom_str:
            continue
        lo = max(ex_start, seq_start) - seq_start
        hi = min(ex_end, seq_end) - seq_start
        if lo >= hi:
            continue
        parts.append(np.arange(lo, hi, dtype=np.int32))

    if not parts:
        return []

    positions = np.unique(np.concatenate(parts))
    if strand < 1:
        positions = seq_len - 1 - positions

    return positions.tolist()


def _predict_one_shift(
    ag_model, x, organism_index, track_bundle, window_bp, device,
    modality, center_index, exon_idx, tta_rev_comp,
):
    """
    Forward (+ optional reverse-complement) prediction for a batch of sequences at
    ONE shift, returning a float64 tensor of shape (B,) averaged over the orientation(s).

    The shift itself is baked into ``x`` by the dataset (each shift is a real-flank
    window re-fetched at ``seq_start + shift``; no rolling), and ``center_index`` /
    ``exon_idx`` are that shift's readout indices. The reverse-complement augmentation
    is a tensor flip here: ``torch.flip(x, [1, 2])`` read on the **minus-strand**
    tracks at flipped indices ``L-1-idx``.
    """
    L = x.shape[1]
    if isinstance(track_bundle, dict):
        plus_inds = track_bundle["plus"]
        minus_inds = track_bundle["minus"]
    else:  # plain list (strand-unaware) — same indices for both orientations
        plus_inds = minus_inds = track_bundle

    sig = _forward_once(ag_model, x, organism_index, plus_inds, device, modality)
    acc = torch.tensor(
        _aggregate_sig(sig, modality, window_bp, center_index, exon_idx), dtype=torch.float64
    )
    n = 1

    if tta_rev_comp:
        x_rc = torch.flip(x, dims=[1, 2])
        rc_center = None if center_index is None else (L - 1) - int(center_index)
        rc_exon = (
            [min(L - 1, max(0, (L - 1) - e)) for e in exon_idx] if exon_idx else exon_idx
        )
        sig = _forward_once(ag_model, x_rc, organism_index, minus_inds, device, modality)
        acc = acc + torch.tensor(
            _aggregate_sig(sig, modality, window_bp, rc_center, rc_exon), dtype=torch.float64
        )
        n += 1

    return acc / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    os.makedirs(args.save_path, exist_ok=True)

    model = load_base_model(args.backbone_model_path, device)
    model_name = "base_alphagenome"
    print("Model name: {}".format(model_name))

    effective_window_bp = (
        ENFORMER_TSS_WINDOW_BP if args.modality == "cage" else args.window_bp
    )

    # ---- Test-time augmentation config --------------------------------------
    tta_shifts = tuple(int(s) for s in args.tta_shifts.split(",") if str(s).strip())
    if not tta_shifts:
        tta_shifts = (0,)
    n_passes = len(tta_shifts) * (2 if args.tta_rev_comp else 1)
    if n_passes > 1:
        print("Test-time augmentation: shifts={} bp (real-flank windows), rev_comp={} "
              "→ {} forward passes/seq".format(list(tta_shifts), args.tta_rev_comp, n_passes))

    # ---- Track selection -----------------------------------------------------
    n_expected = AG_HUMAN_1BP_NUM_TRACKS[args.modality]
    track_inds, track_desc = get_alphagenome_track_indices(
        args.metadata_path,
        args.target_cell_line,
        output_type=args.modality,
        n_total_tracks=n_expected,
    )
    print("\nTrack selection ({}): {}".format(args.modality, track_desc))
    if args.modality == "cage":
        print(
            "Aggregation window: {} bp ({} × {} bp, Enformer-equivalent; --window_bp ignored)".format(
                effective_window_bp, ENFORMER_CENTRAL_BINS, ENFORMER_BIN_BP
            )
        )
    else:
        print(
            "Aggregation window: {:,} bp (1 bp {} resolution)".format(
                effective_window_bp, args.modality
            )
        )

    # ---- Dataset -------------------------------------------------------------
    print("\nLoading Fulco dataset from {}...".format(args.data_path))
    print("Sequence window: centred on ziga TSS (Karollus et al.); enhancer shuffled in place.")
    dataset = FulcoDataset(
        data_path=args.data_path,
        sequence_length=AG_SEQ_LEN,
        N=args.N,
        enhancer_bp=args.enhancer_bp,
        genome_build=args.genome_build,
        validated_only=args.validated_only,
        crispri_perturb_mode=args.crispri_perturb_mode,
        min_enh_dist=args.min_enh_dist,
        gene_tss_csv=args.gene_tss_csv,
        observed_subset=args.fulco_corr_observed_subset,
        tta_shifts=tta_shifts,
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

    # ---- Exon annotations (rna_seq only) ------------------------------------
    exon_coords = None
    if args.modality == "rna_seq":
        gene_ids = dataset.crispri_data["ENSG"].unique().tolist()
        exon_csv = (
            None if str(getattr(args, "exon_csv", "none")).strip().lower() == "none"
            else args.exon_csv
        )
        exon_coords = fetch_or_load_exon_coords(
            gene_ids,
            genome_build=args.genome_build,
            cache_csv=exon_csv,
        )
        if exon_coords is not None:
            n_covered = sum(1 for g in gene_ids if exon_coords.get(g))
            print("Exon coords loaded for {}/{} genes.".format(n_covered, len(gene_ids)))

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # ---- Inference -----------------------------------------------------------
    print("\nRunning inference on {} enhancer-gene pairs...".format(len(dataset)))
    print("  {} dinucleotide shuffles per pair".format(args.N))
    print("  Context: {:,} bp  |  modality: {}  |  shuffle batch size: {}".format(
        AG_SEQ_LEN, args.modality, args.shuffle_batch_size
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
    all_debug_exon_bp = []
    all_debug_track_count = []
    all_debug_track_count_plus = []
    all_debug_track_count_minus = []
    all_debug_crispri_std = []
    all_debug_crispri_min = []
    all_debug_crispri_max = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue

        y_delta = (1.0 - batch["y_delta"]).item()
        strand_i = int(batch["strand"].item()) if torch.is_tensor(batch["strand"]) else int(batch["strand"])
        cur_track_inds = track_inds["plus"] if isinstance(track_inds, dict) else track_inds
        if isinstance(track_inds, dict):
            cur_track_count_plus = len(track_inds.get("plus", []))
            cur_track_count_minus = len(track_inds.get("minus", []))
        else:
            cur_track_count_plus = len(cur_track_inds)
            cur_track_count_minus = len(cur_track_inds)

        # Normalise the batch into a list of per-shift views. The dataset stacks
        # real-flank windows along a leading shift axis when TTA shifts are set;
        # otherwise it returns a single window (legacy shape).
        if "seq_start" in batch and len(tta_shifts) > 1:
            S = batch["x_WT"].shape[0]
            shift_wt = [batch["x_WT"][s : s + 1] for s in range(S)]   # each (1, L, 4)
            shift_cr = [batch["x_crispri"][s] for s in range(S)]      # each (N, L, 4)
            shift_tss = [int(batch["tss_seq_index"][s]) for s in range(S)]
            shift_seqstart = [int(batch["seq_start"][s]) for s in range(S)]
        else:
            shift_wt = [batch["x_WT"]]
            shift_cr = [batch["x_crispri"]]
            shift_tss = [_tss_seq_index_from_batch(batch)]
            shift_seqstart = [int(batch["tss"]) - AG_SEQ_LEN // 2]

        # Per-shift exon readout indices (computed from each shift's window start).
        chrom_val = (
            batch["enh_loc"].split(":")[0]
            if isinstance(batch["enh_loc"], str) else str(batch["enh_loc"])
        )
        gene_id = batch["gene_id"]
        shift_exon = []
        for ss in shift_seqstart:
            if exon_coords is not None and args.modality == "rna_seq":
                shift_exon.append(
                    _exon_seq_indices(gene_id, exon_coords, chrom_val, ss, AG_SEQ_LEN, strand_i)
                )
            else:
                shift_exon.append(None)

        def _one_shift(x_batch, s_i):
            return _predict_one_shift(
                model, x_batch, args.organism_index, track_inds,
                effective_window_bp, device, args.modality,
                shift_tss[s_i], shift_exon[s_i], args.tta_rev_comp,
            )

        # WT: average over shifts (each already averaged over fw/rc inside _one_shift).
        wt_vals = [float(_one_shift(shift_wt[s_i], s_i)[0]) for s_i in range(len(shift_wt))]
        pred_wt = sum(wt_vals) / len(wt_vals)

        # CRISPRi: pool over shifts × shuffles (independent shuffles per shift; the
        # mean is an unbiased estimate of E[pred | shuffled enhancer]).
        pred_crispri_list = []
        bs = args.shuffle_batch_size
        for s_i in range(len(shift_cr)):
            xcr = shift_cr[s_i]  # (N, L, 4)
            for i in range(0, xcr.shape[0], bs):
                pred_crispri_list.extend(_one_shift(xcr[i : i + bs], s_i).tolist())

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
        all_tss_seq_index.append(shift_tss[0])
        all_debug_exon_bp.append(len(shift_exon[0]) if shift_exon[0] else 0)
        all_debug_track_count.append(len(cur_track_inds))
        all_debug_track_count_plus.append(cur_track_count_plus)
        all_debug_track_count_minus.append(cur_track_count_minus)
        all_debug_crispri_std.append(pred_crispri_std)
        all_debug_crispri_min.append(pred_crispri_min)
        all_debug_crispri_max.append(pred_crispri_max)

    if len(all_y_deltas) == 0:
        print("No pairs evaluated -- exiting.")
        return None

    print("\nCompiling results...")
    df_results = pd.DataFrame({
        "gene_id":         all_gene_ids,
        "gene_name":       all_gene_names,
        "enh_loc":         all_enh_locs,
        "tss":             all_tss,
        "tss_seq_index":   all_tss_seq_index,
        "strand":          all_strands,
        "enh_dist":        all_enh_dists,
        "y_delta":         all_y_deltas,
        "pred_delta":      all_pred_deltas,
        "pred_wt":         all_pred_wt,
        "pred_crispri_mean": all_pred_crispri_mean,
    })
    if args.debug_export:
        df_results["debug_exon_bp_used"] = all_debug_exon_bp
        df_results["debug_track_count_used"] = all_debug_track_count
        df_results["debug_track_count_plus"] = all_debug_track_count_plus
        df_results["debug_track_count_minus"] = all_debug_track_count_minus
        df_results["debug_crispri_std"] = all_debug_crispri_std
        df_results["debug_crispri_min"] = all_debug_crispri_min
        df_results["debug_crispri_max"] = all_debug_crispri_max

    # Resolve output prefix (shared by shard CSVs and the full summary).
    if args.output_prefix is None:
        output_prefix = "Fulco_AlphaGenome_base_{}".format(args.modality)
    else:
        output_prefix = args.output_prefix

    # In sharded mode, write only this shard's per-pair CSV; the merge step
    # (scripts/merge_tta_shards.py) concatenates shards and computes correlations.
    if args.num_shards > 1:
        shard_csv = os.path.join(
            args.save_path,
            "{}_shard{:03d}of{:03d}_results.csv".format(
                output_prefix, args.shard_idx, args.num_shards
            ),
        )
        df_results.to_csv(shard_csv, index=False)
        print("\nShard {}/{} results saved to: {}".format(
            args.shard_idx, args.num_shards, shard_csv))
        print("Merge with: python scripts/merge_tta_shards.py "
              "--shards '{}/{}_shard*of{:03d}_results.csv' --output_prefix {} "
              "--fulco_corr_observed_subset {}".format(
                  args.save_path.rstrip("/"), output_prefix, args.num_shards,
                  output_prefix, args.fulco_corr_observed_subset))
        return df_results

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
    print(
        "Linear fractional (y_delta vs pred_delta): not comparable scale for RNA-seq — "
        "pred_delta is often ~1e-4 while y_delta is ~0.01–1; interpret cautiously."
    )
    print("Pearson correlation:  {:.4f}  (p={:.2e})".format(pearson_r, pearson_p))
    print("Spearman correlation: {:.4f}  (p={:.2e})".format(spearman_r, spearman_p))
    print("Log2 knockdown alignment (-log2(expr_ratio) vs log2(pred_wt/pred_crispri)):")
    print("Pearson correlation:  {:.4f}  (p={:.2e})".format(log_pear_r, log_pear_p))
    print("Spearman correlation: {:.4f}  (p={:.2e})".format(log_spr_r, log_spr_p))
    print("=" * 80)

    csv_path = os.path.join(args.save_path, "{}_results.csv".format(output_prefix))
    df_results.to_csv(csv_path, index=False)
    print("\nResults saved to: {}".format(csv_path))

    summary_path = os.path.join(args.save_path, "{}_summary.txt".format(output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULCO BENCHMARK SUMMARY -- AlphaGenome\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write("Model: base AlphaGenome\n")
        f.write("Backbone:   {}\n".format(args.backbone_model_path or "HuggingFace gtca/alphagenome_pytorch"))
        f.write("\nEVALUATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("Context window:    {:,} bp\n".format(AG_SEQ_LEN))
        f.write("Genome build:      {}\n".format(args.genome_build))
        f.write("Fulco data path:   {}\n".format(args.data_path))
        f.write("Gene TSS CSV:      {}\n".format(args.gene_tss_csv))
        f.write("Validated only:    {}\n".format(args.validated_only))
        f.write("N shuffles:        {}\n".format(args.N))
        f.write("Enhancer window:   {} bp\n".format(args.enhancer_bp))
        f.write("Modality (1 bp head): {}\n".format(args.modality))
        f.write("TTA shifts:        {} bp\n".format(list(tta_shifts)))
        f.write("TTA reverse-comp:  {}\n".format(args.tta_rev_comp))
        f.write("TTA passes/seq:    {}\n".format(n_passes))
        f.write("Target cell line:  {}\n".format(args.target_cell_line))
        if args.metadata_path:
            f.write("Metadata file:     {}\n".format(args.metadata_path))
        f.write("Tracks:            {}\n".format(track_desc))
        if args.modality == "rna_seq" and exon_coords is not None:
            f.write("Aggregation:       mean over GENCODE exon positions (AlphaGenome paper approach)\n")
        else:
            f.write(
                "Aggregation window:{:,} bp (summed around TSS index)\n".format(effective_window_bp)
            )
        f.write("Scoring:           fractional drop (pred_WT - pred_CRISPRi) / (pred_WT + eps)\n")
        if args.modality == "cage":
            f.write(
                "Enformer match:    {} central bins × {} bp = {} bp\n".format(
                    ENFORMER_CENTRAL_BINS, ENFORMER_BIN_BP, ENFORMER_TSS_WINDOW_BP
                )
            )
        f.write("Resolution:        1 bp (full AlphaGenome forward)\n")
        f.write(
            "Fulco correlation observed subset: {}\n".format(args.fulco_corr_observed_subset)
        )
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Pairs evaluated: {}\n".format(n_full))
        f.write("Pairs used for Pearson/Spearman: {}\n".format(n_corr))
        f.write(
            "Pearson/Spearman (linear y_delta vs pred_delta): see script note — scales often incomparable.\n"
        )
        f.write("Pearson r:       {:.4f}  (p={:.2e})\n".format(pearson_r, pearson_p))
        f.write("Spearman r:      {:.4f}  (p={:.2e})\n".format(spearman_r, spearman_p))
        f.write("Log2 alignment (-log2(expr_ratio) vs log2(pred_wt/pred_crispri); primary for RNA-seq):\n")
        f.write("Pearson r:       {:.4f}  (p={:.2e})\n".format(log_pear_r, log_pear_p))
        f.write("Spearman r:      {:.4f}  (p={:.2e})\n".format(log_spr_r, log_spr_p))
        f.write("=" * 80 + "\n")

    print("Summary saved to: {}".format(summary_path))
    return df_results


if __name__ == "__main__":
    main()
