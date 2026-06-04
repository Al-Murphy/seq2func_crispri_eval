#!/usr/bin/env python3
"""
Test base Nucleotide Transformer V3 (InstaDeepAI/NTv3_650M_post) on Fulco et al.
(2019) CRISPRi enhancer knockdown pairs (Karollus et al. 2023 windowing).

Base-model evaluation only. See README for adding new models.

NTv3_650M_post is a conditioned-U-Net DNA model post-trained on ~7,362 human
BigWig regression tracks plus BED element heads. It accepts character-tokenised
DNA up to 1 Mb in length and returns ``bigwig_tracks_logits`` cropped to the
middle 37.5 % of the input window (1 bp resolution).

For the BASE model:
  - K562 RNA-Seq / CAGE default to the full set of K562 tracks NTv3 shares with
    Borzoi's targets_human.txt panel (RNA-Seq: 86 track columns from 67
    accessions; CAGE: CNhs11250 + CNhs12334 strand pairs = 4 columns). DNase /
    GATA1 default to the single IDs from the official NTv3 demo notebook
    (InstaDeepAI/ntv3 Space, ``05_model_interpretation.ipynb``):
        RNA-Seq    -> 86 K562 tracks shared with Borzoi
        CAGE       -> CNhs11250 + CNhs12334 strand pairs (4 tracks)
        DNase      -> ENCSR921NMD
        GATA1 ChIP -> ENCSR000EFT
    Override these with ``--track_names``, ``--track_names_csv``, or
    ``--track_indices``. ``--use_borzoi_xref`` reproduces the RNA-Seq / CAGE
    defaults by cross-referencing Borzoi's track list at runtime.
  - CAGE / DNase / GATA1: signal summed over a TSS-centred bp window and the
    selected K562 tracks (Borzoi-style track-sum). RNA-Seq: AlphaGenome-style
    exon aggregation -- mean over GENCODE exon bp (mapped into NTv3's central
    output crop), then mean over tracks (see --exon_csv).
  - Experimental labels match Enformer/AlphaGenome/Borzoi:
    ``y_delta = 1 - expr_ratio`` (higher = stronger repression vs WT).
  - IMPORTANT: NTv3's ``bigwig_tracks_logits`` field name is misleading. Per
    the paper (Section 4.3.3) and the LinearHead source, the BigWig head ends
    with ``F.softplus``, so outputs are already **non-negative** signal in
    normalised-CPM space (per-track mean normalisation; RNA-seq tracks
    additionally get an x^0.75 squash + smooth clip, see paper Section 4.6.1).
    Do NOT apply Softplus again -- it would distort the dynamic range of
    low-signal predictions. ``--apply_softplus`` is therefore off by default
    and exposed only for debugging.
  - Default scoring (``--score_mode linear_frac``) matches the
    Borzoi/AlphaGenome fractional-drop convention so the four models can be
    plotted on a comparable scale:
    ``pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)``.
    The per-track normalisation factor (mu_t for NTv3) cancels in this ratio,
    so the squash on RNA-seq tracks only affects the absolute pred_delta
    scale, not the per-pair ranking / correlation.
  - ``--score_mode log2_fc`` reproduces the NTv3 paper's standard variant
    scoring (Section 4.10.2): ``log2((pred_wt + 1e-6) / (pred_crispri + 1e-6))``.

NTv3 specifics:
  - Input length must be a multiple of 128 (tokenizer pads to 128).
  - Output covers genomic span ``[tss - L/2 + 0.3125 L, tss - L/2 + 0.6875 L)``
    in input-window coordinates -- so when the input is centred on the TSS,
    the TSS lands at output index ``L_pred // 2``.
  - Species token: ``model.encode_species(["human"])`` (matches ``cfg.species_to_token_id``).

Usage:
    # K562 RNA-Seq (ENCSR056HPM), 1 Mb context (defaults)
    python scripts/test_fulco_ntv3.py \\
        --data_path ./metadata/fulco/ \\
        --gene_tss_csv ./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv \\
        --autocast_dtype bf16

    # K562 CAGE (CNhs11250 strand pair)
    python scripts/test_fulco_ntv3.py --target_assay CAGE

    # Custom track set (e.g. multiple K562 RNA-Seq ENCSRs)
    python scripts/test_fulco_ntv3.py \\
        --track_names ENCSR056HPM,ENCSR000AEP,ENCSR000AEQ

    # Aggregate every Borzoi K562 RNA-Seq track that overlaps NTv3 (~48 tracks)
    python scripts/test_fulco_ntv3.py --use_borzoi_xref
"""

import argparse
import math
import os
import re
import warnings
import json
import urllib.request as _urlreq

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from crispri_eval.datasets import FulcoDataset
from crispri_eval.benchmark_utils import fulco_results_for_correlation


# NTv3_650M_post constants
NTV3_DEFAULT_MODEL = "InstaDeepAI/NTv3_650M_post"
NTV3_SEQ_LEN = 1_048_576           # input bp (max context; multiple of 128)
NTV3_PAD_MULTIPLE = 128
# Output covers the middle 37.5% of the input window (config: keep_target_center_fraction)
NTV3_PRED_CENTER_FRACTION = 0.375
NTV3_PRED_OFFSET_FRACTION = 0.3125  # 0.5 * (1 - 0.375)

# Default TSS-centred aggregation: match Enformer 5 x 128 bp = 640 bp window.
DEFAULT_WINDOW_BP = 640

# NTv3 vocab (A=6, T=7, C=8, G=9, N=10); FulcoDataset one-hot channel order is ACGT.
# Build a (4,) tensor mapping one-hot argmax -> NTv3 token id at the chars [A,C,G,T].
NTV3_ACGT_TO_TOKEN = torch.tensor([6, 8, 9, 7], dtype=torch.long)
NTV3_N_TOKEN_ID = 10

# K562 track IDs from the InstaDeepAI/ntv3 demo notebook
# (Space: notebooks_tutorials/05_model_interpretation.ipynb).
# Verified present in cfg.bigwigs_per_species["human"] for NTv3_650M_post.
# K562 RNA-Seq default: the K562 RNA accessions NTv3 shares with Borzoi's
# targets_human.txt panel (67 of Borzoi's 71 unique K562 RNA accessions ->
# 86 NTv3 track columns after _P/_M strand expansion). NTv3 was post-trained on
# Borzoi's track set, so this reproduces Borzoi's K562 RNA panel. Predictions are
# SUMMED over these tracks (see _aggregate), matching Borzoi's track-sum
# convention. Generated via the Borzoi xref
# (see --use_borzoi_xref); override with --track_names for a single track.
K562_RNA_TRACKS = [
    'ENCSR000AEL_M', 'ENCSR000AEL_P', 'ENCSR000AEM_M', 'ENCSR000AEM_P',
    'ENCSR000AEN_M', 'ENCSR000AEN_P', 'ENCSR000AEP', 'ENCSR000AEQ',
    'ENCSR000CPY_M', 'ENCSR000CPY_P', 'ENCSR000CPZ_M', 'ENCSR000CPZ_P',
    'ENCSR000CQA_M', 'ENCSR000CQA_P', 'ENCSR000CWG', 'ENCSR000CWH',
    'ENCSR000CWI', 'ENCSR000CWJ', 'ENCSR000EYO', 'ENCSR006EBD',
    'ENCSR038WEK_M', 'ENCSR038WEK_P', 'ENCSR040YBR_M', 'ENCSR040YBR_P',
    'ENCSR056HPM', 'ENCSR062FHL', 'ENCSR100JNS', 'ENCSR100VUY',
    'ENCSR105NQB', 'ENCSR109IQO_M', 'ENCSR109IQO_P', 'ENCSR114LNC',
    'ENCSR115PIZ', 'ENCSR124KOZ', 'ENCSR161RSX', 'ENCSR165EQJ',
    'ENCSR195JRH', 'ENCSR206KFV', 'ENCSR223DWL', 'ENCSR264IXQ',
    'ENCSR287DHQ', 'ENCSR303IRE', 'ENCSR325BJP', 'ENCSR340QZY',
    'ENCSR369MDF', 'ENCSR379BAF', 'ENCSR384ZXD_M', 'ENCSR384ZXD_P',
    'ENCSR451LDB', 'ENCSR492KRY', 'ENCSR518BSQ', 'ENCSR530NHO_M',
    'ENCSR530NHO_P', 'ENCSR563ZWI', 'ENCSR594NJP_M', 'ENCSR594NJP_P',
    'ENCSR596ACL_M', 'ENCSR596ACL_P', 'ENCSR601DZY', 'ENCSR615EEK_M',
    'ENCSR615EEK_P', 'ENCSR637VLS_M', 'ENCSR637VLS_P', 'ENCSR672NDZ',
    'ENCSR689NPY', 'ENCSR693JOK', 'ENCSR696YIB_M', 'ENCSR696YIB_P',
    'ENCSR733QST', 'ENCSR738ZHN', 'ENCSR758BAT', 'ENCSR792OIJ_M',
    'ENCSR792OIJ_P', 'ENCSR806HCA', 'ENCSR810ZKJ', 'ENCSR844RSF',
    'ENCSR860DWK_M', 'ENCSR860DWK_P', 'ENCSR866UOA', 'ENCSR882NWV',
    'ENCSR885DVH_M', 'ENCSR885DVH_P', 'ENCSR900IJI', 'ENCSR949BBZ',
    'ENCSR950YTM', 'ENCSR957GVE',
]

# K562 CAGE default: both K562 CAGE strand pairs (CNhs11250 + CNhs12334),
# matching Borzoi's 4 K562 CAGE tracks.
K562_CAGE_TRACKS = ['CNhs11250_P', 'CNhs11250_M', 'CNhs12334_P', 'CNhs12334_M']

K562_DEMO_TRACKS = {
    "RNA":   K562_RNA_TRACKS,
    "CAGE":  K562_CAGE_TRACKS,
    "DNase": ["ENCSR921NMD"],
    "GATA1": ["ENCSR000EFT"],
}


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def _tss_output_index(tss_seq_index, seq_len=NTV3_SEQ_LEN):
    """Map an input-sequence index (0..L-1) to an NTv3 output index.

    NTv3 crops the output to the middle 37.5 % of the input, so the output
    bp range in input coordinates is ``[L*0.3125, L*0.6875)``. The output
    index for a TSS at ``tss_seq_index`` (input-space) is therefore
    ``tss_seq_index - int(L * 0.3125)``.
    """
    pred_offset = int(seq_len * NTV3_PRED_OFFSET_FRACTION)
    pred_len = int(seq_len * NTV3_PRED_CENTER_FRACTION)
    idx = int(tss_seq_index) - pred_offset
    if idx < 0 or idx >= pred_len:
        warnings.warn(
            "TSS at sequence index {} falls outside NTv3's output crop "
            "[{}, {}); clamping the readout to the nearest edge.".format(
                tss_seq_index, pred_offset, pred_offset + pred_len
            )
        )
    return int(max(0, min(pred_len - 1, idx))), pred_len


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Test base NTv3 on Fulco CRISPRi benchmark (Karollus windows)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model ----------------------------------------------------------------
    p.add_argument(
        "--model_name",
        type=str,
        default=NTV3_DEFAULT_MODEL,
        help=(
            "HuggingFace model id (or local snapshot path) for the post-trained "
            "NTv3 checkpoint. Loaded via "
            "AutoModel.from_pretrained(..., trust_remote_code=True)."
        ),
    )
    p.add_argument(
        "--species",
        type=str,
        default="human",
        help=(
            "Species token used for the conditioned U-Net. Must be one of "
            "cfg.bigwigs_per_species (e.g. 'human', 'mouse')."
        ),
    )

    # -- Track selection ------------------------------------------------------
    p.add_argument(
        "--target_assay",
        type=str,
        default="RNA",
        choices=["RNA", "CAGE", "DNase", "GATA1"],
        help=(
            "Pick the canonical K562 track set from the InstaDeepAI/ntv3 demo "
            "notebook. RNA -> 86 K562 tracks shared with Borzoi; CAGE -> CNhs11250 + CNhs12334 (4 tracks); "
            "DNase -> ENCSR921NMD; GATA1 -> ENCSR000EFT. Override with "
            "--track_names / --track_names_csv / --track_indices / --use_borzoi_xref."
        ),
    )
    p.add_argument(
        "--target_cell_line",
        type=str,
        default="K562",
        help=(
            "Cell line label (used only for Borzoi cross-reference and result "
            "naming). The hard-coded K562 demo IDs are K562-specific."
        ),
    )
    p.add_argument(
        "--track_names",
        type=str,
        default=None,
        help=(
            "Comma-separated NTv3 track names (matching cfg.bigwigs_per_species[species] "
            "entries, e.g. 'ENCSR056HPM,ENCSR000AEP'). Overrides --target_assay defaults."
        ),
    )
    p.add_argument(
        "--track_names_csv",
        type=str,
        default=None,
        help=(
            "Optional one-column txt/csv listing NTv3 track names. Overrides "
            "--track_names and --target_assay. Header optional."
        ),
    )
    p.add_argument(
        "--track_indices",
        type=str,
        default=None,
        help=(
            "Comma-separated integer indices into cfg.bigwigs_per_species[species] "
            "(overrides every other track-selection arg)."
        ),
    )
    p.add_argument(
        "--use_borzoi_xref",
        action="store_true",
        help=(
            "Aggregate every NTv3 track whose accession appears in Borzoi's K562 "
            "RNA / CAGE metadata. ~48 K562 RNA-Seq ENCSRs overlap; K562 CAGE "
            "CNhs IDs only overlap when the _P/_M strand suffix is included."
        ),
    )
    p.add_argument(
        "--borzoi_targets_csv",
        type=str,
        default="./metadata/borzoi_human_targets.txt",
        help=(
            "Path to Borzoi's targets_human.txt (only used with --use_borzoi_xref). "
            "Cross-references Borzoi's human-readable cell-line / assay descriptions "
            "against NTv3's ENCSR/CNhs/GSM/GTEX track list."
        ),
    )
    p.add_argument(
        "--window_bp",
        type=int,
        default=DEFAULT_WINDOW_BP,
        help=(
            "TSS-centred window in bp summed over the selected tracks. "
            "Default {} bp matches Enformer's 5 x 128 bp central window.".format(
                DEFAULT_WINDOW_BP
            )
        ),
    )
    p.add_argument(
        "--apply_softplus",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=False,
        help=(
            "DEBUG-ONLY: apply a second Softplus to bigwig_tracks_logits. "
            "The NTv3 BigWig head already ends in Softplus (paper Section 4.3.3 "
            "and LinearHead source), so outputs are non-negative signal -- "
            "applying Softplus a second time only distorts low-signal values. "
            "Leave this at False unless you have a specific reason to enable it."
        ),
    )
    p.add_argument(
        "--score_mode",
        choices=["linear_frac", "log2_fc", "log_diff"],
        default="linear_frac",
        help=(
            "How to compute pred_delta from per-pair pred_wt / pred_crispri_mean. "
            "'linear_frac' (default): (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8) "
            "-- matches the Borzoi/AlphaGenome fractional-drop convention so all "
            "four models share a plotting scale. "
            "'log2_fc': log2((pred_wt + 1e-6) / (pred_crispri_mean + 1e-6)) -- "
            "matches the NTv3 paper's standard variant scoring (Section 4.10.2; "
            "epsilon also 1e-6 as in the paper). "
            "'log_diff': pred_wt - pred_crispri_mean (raw difference; not bounded)."
        ),
    )

    # -- Inference batching ---------------------------------------------------
    p.add_argument(
        "--shuffle_batch_size",
        type=int,
        default=1,
        help=(
            "CRISPRi shuffles per forward pass. NTv3 + 1 Mb input is memory "
            "heavy (~5.8 GB per sample at bf16 for the full 393k x 7362 output). "
            "Use 1 unless on a >40 GB GPU."
        ),
    )

    # -- Data -----------------------------------------------------------------
    p.add_argument(
        "--data_path",
        type=str,
        default="./metadata/fulco/",
        help="Directory with ziga_additional_columns.tsv and enhancer_knockdown_effects.tsv.",
    )
    p.add_argument(
        "--gene_tss_csv",
        type=str,
        default="./metadata/Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv",
        help="Biomart CSV with Gene stable ID / Strand; same format as Gasperini TSS export.",
    )
    p.add_argument("--genome_build", type=str, default="hg38", choices=["hg38", "hg19"])
    p.add_argument("--N", type=int, default=50,
                   help="Number of dinucleotide shuffles per enhancer.")
    p.add_argument("--enhancer_bp", type=int, default=2000,
                   help="Size of enhancer region to shuffle (bp).")
    p.add_argument(
        "--crispri_perturb_mode",
        type=str,
        default="dinucleotide",
        choices=["dinucleotide", "shuffle_bases", "random_permute"],
        help=(
            "CRISPRi negatives: dinucleotide (default); shuffle_bases — same bases "
            "in the enhancer, order uniformly shuffled (alias: random_permute)."
        ),
    )
    p.add_argument(
        "--validated_only",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="If True, keep only validated==True rows in ziga (default: True).",
    )
    p.add_argument(
        "--min_enh_dist",
        type=int,
        default=990,
        metavar="BP",
        help=(
            "Minimum enhancer-TSS distance (bp, exclusive). Pairs with enh_dist <= this "
            "value are excluded. SequenceModelBenchmark uses > 990 bp."
        ),
    )
    p.add_argument(
        "--fulco_corr_observed_subset",
        type=str,
        default="knockdown_only",
        choices=["all", "knockdown_only"],
        help=(
            "Rows for Pearson/Spearman vs pred_delta: 'all' — every evaluated pair; "
            "'knockdown_only' — y_delta > 0 only (measured RNA decrease)."
        ),
    )
    p.add_argument(
        "--gene",
        type=str,
        default=None,
        help="Optional: restrict evaluation to pairs for this 'target_gene_short'.",
    )

    # -- Output ---------------------------------------------------------------
    p.add_argument(
        "--save_path",
        type=str,
        default="./results/test_Fulco_ntv3/",
        help="Output directory.",
    )
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument(
        "--exon_csv",
        type=str,
        default="./metadata/Fulco_gene_exons_hg38.csv",
        help=(
            "CSV of GENCODE exon intervals [gene_id, chrom, start, end] (0-based "
            "half-open) for RNA-Seq exon aggregation: mean over exon bp, then over "
            "tracks (AlphaGenome-style), with exon positions mapped into NTv3's "
            "central output crop. Auto-fetched via Ensembl REST if absent. Pass "
            "'none' to fall back to the TSS-centred --window_bp sum. RNA only."
        ),
    )
    p.add_argument(
        "--debug_export",
        action="store_true",
        help="Include per-pair diagnostics (track count, WT/shuffle spread) in the CSV.",
    )

    # -- Runtime --------------------------------------------------------------
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--autocast_dtype",
        type=str,
        default=None,
        choices=[None, "bf16", "fp16"],
        help=(
            "Run forward passes under torch.autocast(dtype=...) for speed/memory. "
            "Recommended: bf16 on modern NVIDIA GPUs."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# K562 track resolution
# ---------------------------------------------------------------------------

def _lookup_names(names, name_to_idx, source_desc):
    """Resolve a list of NTv3 track names to indices, warning on misses."""
    idx, missing = [], []
    for n in names:
        if n in name_to_idx:
            idx.append(name_to_idx[n])
        else:
            missing.append(n)
    if not idx:
        raise ValueError(
            "None of the tracks in {} matched NTv3's track list: {}".format(
                source_desc, names[:5]
            )
        )
    if missing:
        warnings.warn(
            "{} / {} tracks from {} not found in NTv3; first missing: {}".format(
                len(missing), len(names), source_desc, missing[:5]
            )
        )
    return idx


def _borzoi_xref(borzoi_targets_csv, cell_line, assay, ntv3_track_names, name_to_idx):
    """Cross-reference Borzoi's K562 metadata against NTv3's track list.

    NTv3 names CAGE tracks as ``CNhs<id>_P`` / ``CNhs<id>_M`` (strand suffix),
    while Borzoi uses the bare ``CNhs<id>``; we emit both forms when xref-ing
    CAGE so the strand pair is recovered.
    """
    if not os.path.exists(borzoi_targets_csv):
        raise FileNotFoundError(
            "Borzoi targets file not found: {}\nDownload from calico/borzoi "
            "(examples/targets_human.txt) or pass --track_names / --track_indices.".format(
                borzoi_targets_csv
            )
        )

    df = pd.read_csv(borzoi_targets_csv, sep="\t")
    desc_prefix = "{}:".format(assay)
    mask = (
        df["description"].astype(str).str.startswith(desc_prefix)
        & df["description"].astype(str).str.contains(cell_line, case=False, na=False)
    )
    if mask.sum() == 0:
        raise ValueError(
            "No Borzoi tracks matched assay='{}' cell_line='{}'.".format(assay, cell_line)
        )

    pat = re.compile(r"(ENCSR\w+|CNhs\d+|GSM\d+)")
    accessions = (
        df.loc[mask, "file"]
        .astype(str)
        .map(lambda s: pat.search(s).group(1) if pat.search(s) else None)
        .dropna()
    )
    unique_acc = sorted(set(accessions))

    # NTv3 CAGE tracks carry _P/_M strand suffix; Borzoi gives the bare CNhs id.
    candidates = []
    for acc in unique_acc:
        if acc.startswith("CNhs"):
            candidates.extend([acc + "_P", acc + "_M", acc])
        else:
            candidates.append(acc)

    idx = sorted({name_to_idx[c] for c in candidates if c in name_to_idx})
    matched_names = [ntv3_track_names[i] for i in idx]
    return idx, unique_acc, matched_names


def _resolve_track_indices(args, ntv3_track_names):
    """
    Resolve a list of NTv3 track indices to aggregate.

    Resolution order (highest priority first):
        1. --track_indices (explicit integers)
        2. --track_names_csv (one name per line)
        3. --track_names (comma-separated names on CLI)
        4. --use_borzoi_xref (Borzoi target xref for K562 RNA / CAGE)
        5. K562_DEMO_TRACKS[target_assay] (default: the IDs called out in the
           InstaDeepAI/ntv3 05_model_interpretation.ipynb demo)
    """
    name_to_idx = {n: i for i, n in enumerate(ntv3_track_names)}

    if args.track_indices:
        idx = [int(s) for s in args.track_indices.split(",") if s.strip()]
        return idx, "{} explicit indices".format(len(idx))

    if args.track_names_csv:
        names = pd.read_csv(args.track_names_csv, header=None).iloc[:, 0].astype(str).tolist()
        if names and names[0].lower() in ("track_name", "name", "id"):
            names = names[1:]
        idx = _lookup_names(names, name_to_idx, args.track_names_csv)
        return idx, "{} NTv3 tracks (from {})".format(len(idx), args.track_names_csv)

    if args.track_names:
        names = [n.strip() for n in args.track_names.split(",") if n.strip()]
        idx = _lookup_names(names, name_to_idx, "--track_names")
        return idx, "{} NTv3 tracks (--track_names: {})".format(
            len(idx), ", ".join(names[:4]) + ("..." if len(names) > 4 else "")
        )

    if args.use_borzoi_xref:
        idx, unique_acc, matched = _borzoi_xref(
            args.borzoi_targets_csv, args.target_cell_line, args.target_assay,
            ntv3_track_names, name_to_idx,
        )
        if not idx:
            raise ValueError(
                "Borzoi xref produced 0 NTv3 indices for {} {}. NTv3 K562 CAGE "
                "uses CNhs<id>_P/_M strand suffixes; the bare CNhs ids in Borzoi "
                "are not in NTv3 even with suffix expansion. Try --target_assay "
                "RNA or pass --track_names.".format(args.target_cell_line, args.target_assay)
            )
        print("[track xref] {} accessions from Borzoi ({} {} mask) -> {} NTv3 tracks.".format(
            len(unique_acc), args.target_cell_line, args.target_assay, len(idx)
        ))
        return idx, "{} NTv3 tracks (Borzoi xref: {} {})".format(
            len(idx), args.target_cell_line, args.target_assay
        )

    # Default: K562 demo tracks
    if args.target_assay not in K562_DEMO_TRACKS:
        raise ValueError(
            "No demo defaults for target_assay='{}'. Use --track_names / "
            "--track_names_csv / --track_indices or --use_borzoi_xref.".format(args.target_assay)
        )
    names = K562_DEMO_TRACKS[args.target_assay]
    idx = _lookup_names(names, name_to_idx, "K562 demo {}".format(args.target_assay))
    return idx, "{} K562 {} demo tracks: {}".format(len(idx), args.target_assay, ", ".join(names))


# ---------------------------------------------------------------------------
# One-hot -> NTv3 token id conversion
# ---------------------------------------------------------------------------

def onehot_to_ntv3_tokens(x_onehot, device):
    """
    Convert (B, L, 4) one-hot DNA (ACGT order, all-zeros = N) to NTv3 token IDs
    (B, L) on *device*.

    NTv3 vocab: A=6, T=7, C=8, G=9, N=10. FulcoDataset / GasperiniDataset use
    ACGT channel order, so ``argmax(dim=-1)`` -> {0,1,2,3} = {A,C,G,T}; we map
    that into NTv3 token IDs via ``NTV3_ACGT_TO_TOKEN``. Positions where the
    one-hot is all zeros (N / padding) are remapped to the N token (10).
    """
    if x_onehot.dim() != 3 or x_onehot.shape[-1] != 4:
        raise ValueError("Expected (B, L, 4) one-hot DNA; got {}".format(tuple(x_onehot.shape)))
    x = x_onehot.to(device, non_blocking=True)
    lut = NTV3_ACGT_TO_TOKEN.to(device)
    argmax = x.argmax(dim=-1)             # (B, L)
    tokens = lut[argmax]                  # (B, L)
    # Detect N positions (no one-hot bit set)
    is_n = x.sum(dim=-1) <= 0.0
    if is_n.any():
        tokens = torch.where(is_n, torch.full_like(tokens, NTV3_N_TOKEN_ID), tokens)
    return tokens


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ntv3(model_name, device):
    """Load the post-trained NTv3 model + tokenizer with trust_remote_code=True."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    print("Loading NTv3 from: {}".format(model_name))
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(device)
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _autocast_ctx(device, autocast_dtype):
    if autocast_dtype is None or device == "cpu":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if autocast_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _aggregate(pred, track_inds, tss_output_idx, window_bp, apply_softplus, exon_out=None):
    """
    Aggregate NTv3 bigwig_tracks_logits (B, L_pred, T) over the selected
    tracks and a TSS-centred ``window_bp`` window.

    Returns list[float] of length B (sum over the bp window and tracks).
    """
    sig = pred[:, :, track_inds]          # (B, L_pred, n_tracks)
    if apply_softplus:
        sig = torch.nn.functional.softplus(sig)
    if exon_out:
        # RNA-Seq exon aggregation (AlphaGenome-style): mean over exon bp, then
        # over tracks. ``exon_out`` are NTv3 output-crop indices (already mapped).
        idx = torch.as_tensor(exon_out, dtype=torch.long, device=sig.device)
        return sig[:, idx, :].mean(dim=1).mean(dim=1).float().tolist()
    L_pred = sig.shape[1]
    half_lo = window_bp // 2
    half_hi = window_bp - half_lo
    lo = max(0, tss_output_idx - half_lo)
    hi = min(L_pred, tss_output_idx + half_hi)
    sig_window = sig[:, lo:hi, :]
    return sig_window.sum(dim=(1, 2)).float().tolist()


def _predict(model, species_ids_template, x_onehot, track_inds_tensor,
             tss_output_idx, window_bp, apply_softplus, device, autocast_dtype,
             exon_out=None):
    """Forward x through NTv3 and aggregate to a scalar per batch row.

    ``species_ids_template`` is a 1-D long tensor of shape (1,); the model
    expects ``species_ids`` shape (B,), so we expand (not unsqueeze) along the
    batch dim.
    """
    tokens = onehot_to_ntv3_tokens(x_onehot, device)             # (B, L) on device
    B = tokens.shape[0]
    species_ids = species_ids_template[:1].expand(B).contiguous()  # (B,)
    with torch.no_grad(), _autocast_ctx(device, autocast_dtype):
        out = model(input_ids=tokens, species_ids=species_ids)
        pred = out["bigwig_tracks_logits"]                       # (B, L_pred, T)
    return _aggregate(pred, track_inds_tensor, tss_output_idx, window_bp,
                      apply_softplus, exon_out=exon_out)


# ---------------------------------------------------------------------------
# Exon annotation helpers (AlphaGenome-style RNA-Seq scoring, NTv3 crop-mapped)
# ---------------------------------------------------------------------------

def _fetch_exons_from_ensembl(gene_ids, genome_build="hg38"):
    """Fetch exon intervals from the Ensembl REST API.

    Returns dict: {ensg_id: [(chrom_str, start_0based, end_excl), ...]}.
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
    """Return dict {ensg_id: [(chrom, start_0based, end_excl), ...]} or None.

    Loads from *cache_csv* when present (filling any missing genes via the
    Ensembl REST API), otherwise fetches all genes and saves to *cache_csv*.
    Cache format matches the AlphaGenome scripts, so the same
    ``*_gene_exons_hg38.csv`` files are reused.
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
                    len(missing), list(missing)[:3], " ..." if len(missing) > 3 else ""))
            result.update(_fetch_exons_from_ensembl(missing, genome_build))
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

    Shared verbatim with the AlphaGenome scripts: minus-strand sequences are
    reverse-complemented by the dataset, so positions are flipped for strand < 1.
    Returns input-space positions (0..seq_len-1); NTv3 callers then shift these
    into the middle-37.5% output crop.
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
        if lo < hi:
            parts.append(np.arange(lo, hi, dtype=np.int32))
    if not parts:
        return []
    positions = np.unique(np.concatenate(parts))
    if strand < 1:                       # dataset RC's the minus strand
        positions = seq_len - 1 - positions
    return positions.tolist()



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    os.makedirs(args.save_path, exist_ok=True)

    # ---- Model -----------------------------------------------------------
    model, tokenizer, config = load_ntv3(args.model_name, device)
    model_name_tag = re.sub(r"[^\w\-.]+", "_", args.model_name)

    if args.species not in config.bigwigs_per_species:
        raise ValueError(
            "Species '{}' not in cfg.bigwigs_per_species (have {}).".format(
                args.species, list(config.bigwigs_per_species.keys())
            )
        )
    ntv3_track_names = list(config.bigwigs_per_species[args.species])
    print("NTv3 has {} {} BigWig tracks.".format(len(ntv3_track_names), args.species))

    # ---- Track selection -------------------------------------------------
    track_inds, track_desc = _resolve_track_indices(args, ntv3_track_names)
    print("Track selection: {}".format(track_desc))
    track_inds_tensor = torch.as_tensor(track_inds, dtype=torch.long, device=device)

    # ---- Species token ---------------------------------------------------
    species_ids = model.encode_species([args.species])
    species_ids = species_ids.to(device) if torch.is_tensor(species_ids) else torch.as_tensor(
        species_ids, dtype=torch.long, device=device
    )
    if species_ids.dim() != 1:
        species_ids = species_ids.reshape(-1)

    # ---- Dataset ---------------------------------------------------------
    if NTV3_SEQ_LEN % NTV3_PAD_MULTIPLE != 0:
        raise ValueError(
            "NTV3_SEQ_LEN={} must be a multiple of {}.".format(NTV3_SEQ_LEN, NTV3_PAD_MULTIPLE)
        )

    print("\nLoading Fulco dataset from {}...".format(args.data_path))
    print("Sequence window: centred on ziga TSS (Karollus et al.); enhancer shuffled in place.")
    dataset = FulcoDataset(
        data_path=args.data_path,
        sequence_length=NTV3_SEQ_LEN,
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

    # ---- Exon annotations (RNA-Seq exon aggregation) --------------------
    exon_coords = None
    if args.target_assay == "RNA":
        exon_csv = (
            None if str(getattr(args, "exon_csv", "none")).strip().lower() == "none"
            else args.exon_csv
        )
        if exon_csv is not None:
            gene_ids = dataset.crispri_data["ENSG"].unique().tolist()
            exon_coords = fetch_or_load_exon_coords(
                gene_ids, genome_build=args.genome_build, cache_csv=exon_csv,
            )
            if exon_coords is not None:
                n_cov = sum(1 for g in gene_ids if exon_coords.get(g))
                print("Exon coords loaded for {}/{} genes (RNA-Seq exon aggregation).".format(
                    n_cov, len(gene_ids)))
    # NTv3 output-crop offsets (exon input-space positions are shifted into this crop).
    _pred_offset = int(NTV3_SEQ_LEN * NTV3_PRED_OFFSET_FRACTION)
    _pred_len = int(NTV3_SEQ_LEN * NTV3_PRED_CENTER_FRACTION)

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # ---- Inference -------------------------------------------------------
    print("\nRunning inference on {} enhancer-gene pairs...".format(len(dataset)))
    print("  {} dinucleotide shuffles per pair".format(args.N))
    print("  Context: {:,} bp  |  shuffle batch size: {}  |  autocast: {}".format(
        NTV3_SEQ_LEN, args.shuffle_batch_size, args.autocast_dtype or "off"
    ))
    print("  Output bp window covers input [{}, {})".format(
        int(NTV3_SEQ_LEN * NTV3_PRED_OFFSET_FRACTION),
        int(NTV3_SEQ_LEN * NTV3_PRED_OFFSET_FRACTION) + int(NTV3_SEQ_LEN * NTV3_PRED_CENTER_FRACTION),
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
    all_tss_output_index = []
    all_debug_track_count = []
    all_debug_exon_bp = []
    all_debug_crispri_std = []
    all_debug_crispri_min = []
    all_debug_crispri_max = []

    def _predict_batch(x_batch, tss_output_idx, exon_out=None):
        return _predict(
            model, species_ids, x_batch, track_inds_tensor,
            tss_output_idx, args.window_bp, args.apply_softplus,
            device, args.autocast_dtype, exon_out=exon_out,
        )

    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue

        x_wt = batch["x_WT"]            # (1, L, 4) CPU tensor
        x_crispri = batch["x_crispri"]  # (N, L, 4) CPU tensor

        y_delta = (1.0 - batch["y_delta"]).item()

        tss_seq_idx = _tss_seq_index_from_batch(batch)
        tss_output_idx, _ = _tss_output_index(tss_seq_idx, seq_len=NTV3_SEQ_LEN)

        # RNA-Seq: map this gene's GENCODE exons into NTv3's output crop.
        exon_out = None
        if exon_coords is not None:
            tss_i = int(batch["tss"].item()) if torch.is_tensor(batch["tss"]) else int(batch["tss"])
            strand_i = int(batch["strand"].item()) if torch.is_tensor(batch["strand"]) else int(batch["strand"])
            chrom_v = (
                batch["enh_loc"].split(":")[0]
                if isinstance(batch["enh_loc"], str) else str(batch["enh_loc"])
            )
            seq_start = tss_i - NTV3_SEQ_LEN // 2
            exon_in = _exon_seq_indices(
                batch["gene_id"], exon_coords, chrom_v, seq_start, NTV3_SEQ_LEN, strand_i
            )
            exon_out = [p - _pred_offset for p in exon_in if 0 <= p - _pred_offset < _pred_len]
            if not exon_out:
                exon_out = None  # no exon bp landed in the crop -> TSS-window fallback

        pred_wt = _predict_batch(x_wt, tss_output_idx, exon_out)[0]

        pred_crispri_list = []
        bs = max(1, int(args.shuffle_batch_size))
        for i in range(0, x_crispri.shape[0], bs):
            pred_crispri_list.extend(_predict_batch(x_crispri[i : i + bs], tss_output_idx, exon_out))

        pred_crispri_mean = sum(pred_crispri_list) / len(pred_crispri_list)
        pred_crispri_std = float(np.std(pred_crispri_list))
        pred_crispri_min = float(np.min(pred_crispri_list))
        pred_crispri_max = float(np.max(pred_crispri_list))

        if args.score_mode == "linear_frac":
            pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)
        elif args.score_mode == "log2_fc":
            # NTv3 paper Section 4.10.2: log2((alt + eps) / (ref + eps)), eps = 1e-6.
            # Negate so higher pred_delta means stronger repression, matching y_delta = 1-fc.
            pred_delta = -math.log2(
                (pred_crispri_mean + 1e-6) / (pred_wt + 1e-6)
            )
        else:  # log_diff
            pred_delta = pred_wt - pred_crispri_mean

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
        all_tss_output_index.append(tss_output_idx)
        all_debug_track_count.append(len(track_inds))
        all_debug_exon_bp.append(len(exon_out) if exon_out else 0)
        all_debug_crispri_std.append(pred_crispri_std)
        all_debug_crispri_min.append(pred_crispri_min)
        all_debug_crispri_max.append(pred_crispri_max)

    if len(all_y_deltas) == 0:
        print("No pairs evaluated -- exiting.")
        return None

    # ---- Results ---------------------------------------------------------
    print("\nCompiling results...")
    df_results = pd.DataFrame({
        "gene_id":           all_gene_ids,
        "gene_name":         all_gene_names,
        "enh_loc":           all_enh_locs,
        "tss":               all_tss,
        "tss_seq_index":     all_tss_seq_index,
        "tss_output_index":  all_tss_output_index,
        "strand":            all_strands,
        "enh_dist":          all_enh_dists,
        "y_delta":           all_y_deltas,
        "pred_delta":        all_pred_deltas,
        "pred_wt":           all_pred_wt,
        "pred_crispri_mean": all_pred_crispri_mean,
    })
    if args.debug_export:
        df_results["debug_track_count_used"] = all_debug_track_count
        df_results["debug_exon_bp"] = all_debug_exon_bp
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

    # Log2 alignment vs the AlphaGenome/Borzoi convention.
    _eps = 1e-12
    expr_ratio = np.clip(1.0 - df_corr["y_delta"].astype(float).values, _eps, None)
    obs_log_strength = -np.log2(expr_ratio)
    wt_vals = np.maximum(df_corr["pred_wt"].astype(float).values, _eps)
    cr_vals = np.maximum(df_corr["pred_crispri_mean"].astype(float).values, _eps)
    pred_log_strength = np.log2(wt_vals / cr_vals)
    log_pear_r, log_pear_p = pearsonr(obs_log_strength, pred_log_strength)
    log_spr_r, log_spr_p = spearmanr(obs_log_strength, pred_log_strength)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("Pairs evaluated: {}   |   used for correlations: {} (subset={})".format(
        n_full, n_corr, args.fulco_corr_observed_subset
    ))
    print("score_mode={}, apply_softplus={}".format(args.score_mode, args.apply_softplus))
    print("Pearson  (y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(pearson_r, pearson_p))
    print("Spearman (y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(spearman_r, spearman_p))
    print("Log2 alignment (-log2(expr_ratio) vs log2(softplus(WT)/softplus(CRISPRi))):")
    print("  Pearson  r: {:.4f}  (p={:.2e})".format(log_pear_r, log_pear_p))
    print("  Spearman r: {:.4f}  (p={:.2e})".format(log_spr_r, log_spr_p))
    print("=" * 80)

    # ---- Save -----------------------------------------------------------
    if args.output_prefix is None:
        assay_tag = "_{}".format(args.target_assay.lower())
        output_prefix = "Fulco_NTv3_{}{}".format(model_name_tag, assay_tag)
    else:
        output_prefix = args.output_prefix

    csv_path = os.path.join(args.save_path, "{}_results.csv".format(output_prefix))
    df_results.to_csv(csv_path, index=False)
    print("\nResults saved to: {}".format(csv_path))

    summary_path = os.path.join(args.save_path, "{}_summary.txt".format(output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FULCO BENCHMARK SUMMARY -- NTv3\n")
        f.write("=" * 80 + "\n\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write("Model name:        {}\n".format(args.model_name))
        f.write("Species:           {}\n".format(args.species))
        f.write("\nEVALUATION PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("Context window:    {:,} bp\n".format(NTV3_SEQ_LEN))
        f.write("Output crop:       middle {:.1%} of input ({}..{} bp of input window)\n".format(
            NTV3_PRED_CENTER_FRACTION,
            int(NTV3_SEQ_LEN * NTV3_PRED_OFFSET_FRACTION),
            int(NTV3_SEQ_LEN * NTV3_PRED_OFFSET_FRACTION) + int(NTV3_SEQ_LEN * NTV3_PRED_CENTER_FRACTION),
        ))
        f.write("TSS-centred window: {} bp (--window_bp)\n".format(args.window_bp))
        f.write("Apply softplus:    {}\n".format(args.apply_softplus))
        f.write("Score mode:        {}\n".format(args.score_mode))
        f.write("Genome build:      {}\n".format(args.genome_build))
        f.write("Fulco data path:   {}\n".format(args.data_path))
        f.write("Gene TSS CSV:      {}\n".format(args.gene_tss_csv))
        f.write("Validated only:    {}\n".format(args.validated_only))
        f.write("N shuffles:        {}\n".format(args.N))
        f.write("Enhancer window:   {} bp\n".format(args.enhancer_bp))
        f.write("Target cell line:  {}\n".format(args.target_cell_line))
        f.write("Target assay:      {}\n".format(args.target_assay))
        f.write("Tracks:            {}\n".format(track_desc))
        f.write("Fulco subset:      {}\n".format(args.fulco_corr_observed_subset))
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Pairs evaluated:                 {}\n".format(n_full))
        f.write("Pairs used for Pearson/Spearman: {}\n".format(n_corr))
        f.write("Pearson r  (pred_delta):  {:.4f}  (p={:.2e})\n".format(pearson_r, pearson_p))
        f.write("Spearman r (pred_delta):  {:.4f}  (p={:.2e})\n".format(spearman_r, spearman_p))
        f.write("Pearson r  (log2 softplus):  {:.4f}  (p={:.2e})\n".format(log_pear_r, log_pear_p))
        f.write("Spearman r (log2 softplus):  {:.4f}  (p={:.2e})\n".format(log_spr_r, log_spr_p))
        f.write("=" * 80 + "\n")

    print("Summary saved to: {}".format(summary_path))
    return df_results


if __name__ == "__main__":
    main()
