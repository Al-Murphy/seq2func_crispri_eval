#!/usr/bin/env python3
"""
Test base Nucleotide Transformer V3 (InstaDeepAI/NTv3_650M_post) on Gasperini et al.
validated CRISPRi enhancer-gene pairs.

Base-model evaluation only. See README for adding new models.

See ``test_fulco_ntv3.py`` for a long-form explanation of NTv3's output crop,
species token, track-selection options, and ``--score_mode`` semantics.

Usage:
    # K562 RNA-Seq (ENCSR056HPM), 1 Mb context (defaults)
    python scripts/test_gasperini_ntv3.py --autocast_dtype bf16

    # K562 CAGE (CNhs11250 strand pair)
    python scripts/test_gasperini_ntv3.py --target_assay CAGE

    # Custom track set
    python scripts/test_gasperini_ntv3.py \\
        --track_names ENCSR056HPM,ENCSR000AEP,ENCSR000AEQ

    # Aggregate every Borzoi K562 RNA-Seq track that overlaps NTv3 (~48 tracks)
    python scripts/test_gasperini_ntv3.py --use_borzoi_xref
"""

import argparse
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from crispri_eval.datasets import GasperiniDataset


# NTv3_650M_post constants
NTV3_DEFAULT_MODEL = "InstaDeepAI/NTv3_650M_post"
NTV3_SEQ_LEN = 1_048_576           # input bp
NTV3_PAD_MULTIPLE = 128
NTV3_PRED_CENTER_FRACTION = 0.375
NTV3_PRED_OFFSET_FRACTION = 0.3125

DEFAULT_WINDOW_BP = 640

NTV3_ACGT_TO_TOKEN = torch.tensor([6, 8, 9, 7], dtype=torch.long)
NTV3_N_TOKEN_ID = 10

# K562 track IDs from the InstaDeepAI/ntv3 demo notebook.
K562_DEMO_TRACKS = {
    "RNA":   ["ENCSR056HPM"],
    "CAGE":  ["CNhs11250_P", "CNhs11250_M"],
    "DNase": ["ENCSR921NMD"],
    "GATA1": ["ENCSR000EFT"],
}


def _tss_seq_index_from_batch(batch):
    v = batch["tss_seq_index"]
    return int(v.item()) if torch.is_tensor(v) else int(v)


def _tss_output_index(tss_seq_index, seq_len=NTV3_SEQ_LEN):
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Test base NTv3 on Gasperini CRISPRi benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model_name", type=str, default=NTV3_DEFAULT_MODEL,
        help=(
            "HuggingFace model id (or local snapshot path) for the post-trained "
            "NTv3 checkpoint. Loaded via "
            "AutoModel.from_pretrained(..., trust_remote_code=True)."
        ),
    )
    p.add_argument(
        "--species", type=str, default="human",
        help="Species token (must be one of cfg.bigwigs_per_species).",
    )

    p.add_argument(
        "--target_assay", type=str, default="RNA",
        choices=["RNA", "CAGE", "DNase", "GATA1"],
        help=(
            "Pick the canonical K562 track set from the InstaDeepAI/ntv3 demo "
            "notebook. RNA -> ENCSR056HPM; CAGE -> CNhs11250_P + CNhs11250_M; "
            "DNase -> ENCSR921NMD; GATA1 -> ENCSR000EFT. Override with "
            "--track_names / --track_names_csv / --track_indices / --use_borzoi_xref."
        ),
    )
    p.add_argument(
        "--target_cell_line", type=str, default="K562",
        help="Cell line label (used only for Borzoi cross-reference and result naming).",
    )
    p.add_argument(
        "--track_names", type=str, default=None,
        help="Comma-separated NTv3 track names. Overrides --target_assay defaults.",
    )
    p.add_argument(
        "--track_names_csv", type=str, default=None,
        help="One-column txt/csv listing NTv3 track names. Overrides --track_names.",
    )
    p.add_argument(
        "--track_indices", type=str, default=None,
        help="Comma-separated integer indices into cfg.bigwigs_per_species[species].",
    )
    p.add_argument(
        "--use_borzoi_xref", action="store_true",
        help="Aggregate every NTv3 track whose accession appears in Borzoi's K562 RNA / CAGE metadata.",
    )
    p.add_argument(
        "--borzoi_targets_csv", type=str,
        default="./metadata/borzoi_human_targets.txt",
        help="Path to Borzoi's targets_human.txt (only used with --use_borzoi_xref).",
    )
    p.add_argument(
        "--window_bp", type=int, default=DEFAULT_WINDOW_BP,
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
            "Leave at False unless you have a specific reason."
        ),
    )
    p.add_argument(
        "--score_mode", choices=["linear_frac", "log2_fc", "log_diff"],
        default="linear_frac",
        help=(
            "How to compute pred_delta. 'linear_frac' matches Borzoi/AlphaGenome "
            "fractional-drop convention; 'log2_fc' matches NTv3 paper Section 4.10.2; "
            "'log_diff' is raw difference."
        ),
    )

    p.add_argument(
        "--shuffle_batch_size", type=int, default=1,
        help="CRISPRi shuffles per forward pass. NTv3 + 1 Mb input is memory heavy; use 1 unless on a >40 GB GPU.",
    )

    p.add_argument("--data_path", type=str, default="./metadata/")
    p.add_argument("--genome_build", type=str, default="hg38", choices=["hg38", "hg19"])
    p.add_argument("--N", type=int, default=50,
                   help="Number of dinucleotide shuffles per enhancer.")
    p.add_argument("--enhancer_bp", type=int, default=2000,
                   help="Size of enhancer region to shuffle (bp).")
    p.add_argument(
        "--crispri_perturb_mode", type=str, default="dinucleotide",
        choices=["dinucleotide", "shuffle_bases", "random_permute"],
        help=(
            "CRISPRi negatives: dinucleotide (default); shuffle_bases — same bases in "
            "the enhancer, order uniformly shuffled (alias: random_permute)."
        ),
    )
    p.add_argument(
        "--high_confidence_subset",
        type=lambda s: str(s).lower() in ("1", "true", "yes"),
        default=True,
        help="Restrict to high-confidence Gasperini pairs.",
    )
    p.add_argument(
        "--min_enh_dist", type=int, default=990, metavar="BP",
        help=(
            "Minimum enhancer-TSS distance (bp, exclusive). Pairs with enh_dist <= this "
            "value are excluded. SequenceModelBenchmark uses > 990 bp."
        ),
    )
    p.add_argument(
        "--gene", type=str, default=None,
        help="Optional: restrict evaluation to pairs for this 'target_gene_short'.",
    )

    p.add_argument(
        "--save_path", type=str, default="./results/test_Gasperini_ntv3/",
        help="Output directory.",
    )
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument(
        "--debug_export", action="store_true",
        help="Include per-pair diagnostics (track count, WT/shuffle spread) in the CSV.",
    )

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument(
        "--autocast_dtype", type=str, default=None,
        choices=[None, "bf16", "fp16"],
        help="Run forward passes under torch.autocast(dtype=...). Recommended: bf16.",
    )

    return p.parse_args()


def _lookup_names(names, name_to_idx, source_desc):
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

    NTv3 names CAGE tracks as ``CNhs<id>_P`` / ``CNhs<id>_M``; Borzoi uses the
    bare ``CNhs<id>``, so we emit both forms when xref-ing CAGE.
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
    Resolve NTv3 track indices. Priority: --track_indices > --track_names_csv >
    --track_names > --use_borzoi_xref > K562_DEMO_TRACKS[target_assay].
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
                "Borzoi xref produced 0 NTv3 indices for {} {}.".format(
                    args.target_cell_line, args.target_assay
                )
            )
        print("[track xref] {} accessions from Borzoi ({} {} mask) -> {} NTv3 tracks.".format(
            len(unique_acc), args.target_cell_line, args.target_assay, len(idx)
        ))
        return idx, "{} NTv3 tracks (Borzoi xref: {} {})".format(
            len(idx), args.target_cell_line, args.target_assay
        )

    if args.target_assay not in K562_DEMO_TRACKS:
        raise ValueError(
            "No demo defaults for target_assay='{}'.".format(args.target_assay)
        )
    names = K562_DEMO_TRACKS[args.target_assay]
    idx = _lookup_names(names, name_to_idx, "K562 demo {}".format(args.target_assay))
    return idx, "{} K562 {} demo tracks: {}".format(len(idx), args.target_assay, ", ".join(names))


def onehot_to_ntv3_tokens(x_onehot, device):
    """Convert (B, L, 4) one-hot DNA (ACGT order, all-zeros = N) to NTv3 token IDs (B, L)."""
    if x_onehot.dim() != 3 or x_onehot.shape[-1] != 4:
        raise ValueError("Expected (B, L, 4) one-hot DNA; got {}".format(tuple(x_onehot.shape)))
    x = x_onehot.to(device, non_blocking=True)
    lut = NTV3_ACGT_TO_TOKEN.to(device)
    argmax = x.argmax(dim=-1)
    tokens = lut[argmax]
    is_n = x.sum(dim=-1) <= 0.0
    if is_n.any():
        tokens = torch.where(is_n, torch.full_like(tokens, NTV3_N_TOKEN_ID), tokens)
    return tokens


def load_ntv3(model_name, device):
    """Load the post-trained NTv3 model + tokenizer with trust_remote_code=True."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    print("Loading NTv3 from: {}".format(model_name))
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model.eval().to(device)
    return model, tokenizer, config


def _autocast_ctx(device, autocast_dtype):
    if autocast_dtype is None or device == "cpu":
        return torch.amp.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if autocast_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _aggregate(pred, track_inds, tss_output_idx, window_bp, apply_softplus):
    """Aggregate NTv3 bigwig_tracks_logits (B, L_pred, T) over tracks and TSS window."""
    sig = pred[:, :, track_inds]
    if apply_softplus:
        sig = torch.nn.functional.softplus(sig)
    L_pred = sig.shape[1]
    half_lo = window_bp // 2
    half_hi = window_bp - half_lo
    lo = max(0, tss_output_idx - half_lo)
    hi = min(L_pred, tss_output_idx + half_hi)
    sig_window = sig[:, lo:hi, :]
    return sig_window.sum(dim=(1, 2)).float().tolist()


def _predict(model, species_ids_template, x_onehot, track_inds_tensor,
             tss_output_idx, window_bp, apply_softplus, device, autocast_dtype):
    """Forward x through NTv3 and aggregate to a scalar per batch row."""
    tokens = onehot_to_ntv3_tokens(x_onehot, device)
    B = tokens.shape[0]
    species_ids = species_ids_template[:1].expand(B).contiguous()
    with torch.no_grad(), _autocast_ctx(device, autocast_dtype):
        out = model(input_ids=tokens, species_ids=species_ids)
        pred = out["bigwig_tracks_logits"]
    return _aggregate(pred, track_inds_tensor, tss_output_idx, window_bp, apply_softplus)


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    os.makedirs(args.save_path, exist_ok=True)

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

    track_inds, track_desc = _resolve_track_indices(args, ntv3_track_names)
    print("Track selection: {}".format(track_desc))
    track_inds_tensor = torch.as_tensor(track_inds, dtype=torch.long, device=device)

    species_ids = model.encode_species([args.species])
    species_ids = species_ids.to(device) if torch.is_tensor(species_ids) else torch.as_tensor(
        species_ids, dtype=torch.long, device=device
    )
    if species_ids.dim() != 1:
        species_ids = species_ids.reshape(-1)

    if NTV3_SEQ_LEN % NTV3_PAD_MULTIPLE != 0:
        raise ValueError(
            "NTV3_SEQ_LEN={} must be a multiple of {}.".format(NTV3_SEQ_LEN, NTV3_PAD_MULTIPLE)
        )

    print("\nLoading Gasperini dataset from {}...".format(args.data_path))
    print("Sequence window: centred on gene TSS; enhancer shuffled in place.")
    dataset = GasperiniDataset(
        data_path=args.data_path,
        sequence_length=NTV3_SEQ_LEN,
        N=args.N,
        enhancer_bp=args.enhancer_bp,
        genome_build=args.genome_build,
        high_confidence_subset=args.high_confidence_subset,
        crispri_perturb_mode=args.crispri_perturb_mode,
        min_enh_dist=args.min_enh_dist,
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
                "No Gasperini pairs found for gene '{}'.".format(args.gene)
            )

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

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
    all_debug_crispri_std = []
    all_debug_crispri_min = []
    all_debug_crispri_max = []

    def _predict_batch(x_batch, tss_output_idx):
        return _predict(
            model, species_ids, x_batch, track_inds_tensor,
            tss_output_idx, args.window_bp, args.apply_softplus,
            device, args.autocast_dtype,
        )

    for batch in tqdm(dataloader, total=len(dataloader)):
        if batch is None:
            continue

        x_wt = batch["x_WT"]
        x_crispri = batch["x_crispri"]

        y_delta = (1.0 - batch["y_delta"]).item()

        tss_seq_idx = _tss_seq_index_from_batch(batch)
        tss_output_idx, _ = _tss_output_index(tss_seq_idx, seq_len=NTV3_SEQ_LEN)

        pred_wt = _predict_batch(x_wt, tss_output_idx)[0]

        pred_crispri_list = []
        bs = max(1, int(args.shuffle_batch_size))
        for i in range(0, x_crispri.shape[0], bs):
            pred_crispri_list.extend(_predict_batch(x_crispri[i : i + bs], tss_output_idx))

        pred_crispri_mean = sum(pred_crispri_list) / len(pred_crispri_list)
        pred_crispri_std = float(np.std(pred_crispri_list))
        pred_crispri_min = float(np.min(pred_crispri_list))
        pred_crispri_max = float(np.max(pred_crispri_list))

        if args.score_mode == "linear_frac":
            pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)
        elif args.score_mode == "log2_fc":
            pred_delta = -math.log2(
                (pred_crispri_mean + 1e-6) / (pred_wt + 1e-6)
            )
        else:
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
        all_debug_crispri_std.append(pred_crispri_std)
        all_debug_crispri_min.append(pred_crispri_min)
        all_debug_crispri_max.append(pred_crispri_max)

    if len(all_y_deltas) == 0:
        print("No pairs evaluated -- exiting.")
        return None

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
        df_results["debug_crispri_std"] = all_debug_crispri_std
        df_results["debug_crispri_min"] = all_debug_crispri_min
        df_results["debug_crispri_max"] = all_debug_crispri_max

    pearson_r, pearson_p = pearsonr(df_results["y_delta"], df_results["pred_delta"])
    spearman_r, spearman_p = spearmanr(df_results["y_delta"], df_results["pred_delta"])

    _eps = 1e-12
    expr_ratio = np.clip(1.0 - df_results["y_delta"].astype(float).values, _eps, None)
    obs_log_strength = -np.log2(expr_ratio)
    wt_arr = np.maximum(df_results["pred_wt"].astype(float).values, _eps)
    cr_arr = np.maximum(df_results["pred_crispri_mean"].astype(float).values, _eps)
    pred_log_strength = np.log2(wt_arr / cr_arr)
    log_pear_r, log_pear_p = pearsonr(obs_log_strength, pred_log_strength)
    log_spr_r, log_spr_p = spearmanr(obs_log_strength, pred_log_strength)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("Pairs evaluated: {}".format(len(df_results)))
    print("score_mode={}, apply_softplus={}".format(args.score_mode, args.apply_softplus))
    print("Pearson  (y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(pearson_r, pearson_p))
    print("Spearman (y_delta vs pred_delta):  {:.4f}  (p={:.2e})".format(spearman_r, spearman_p))
    print("Log2 alignment (-log2(expr_ratio) vs log2(softplus(WT)/softplus(CRISPRi))):")
    print("  Pearson  r: {:.4f}  (p={:.2e})".format(log_pear_r, log_pear_p))
    print("  Spearman r: {:.4f}  (p={:.2e})".format(log_spr_r, log_spr_p))
    print("=" * 80)

    if args.output_prefix is None:
        assay_tag = "_{}".format(args.target_assay.lower())
        output_prefix = "Gasperini_NTv3_{}{}".format(model_name_tag, assay_tag)
    else:
        output_prefix = args.output_prefix

    csv_path = os.path.join(args.save_path, "{}_results.csv".format(output_prefix))
    df_results.to_csv(csv_path, index=False)
    print("\nResults saved to: {}".format(csv_path))

    summary_path = os.path.join(args.save_path, "{}_summary.txt".format(output_prefix))
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GASPERINI BENCHMARK SUMMARY -- NTv3\n")
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
        f.write("N shuffles:        {}\n".format(args.N))
        f.write("Enhancer window:   {} bp\n".format(args.enhancer_bp))
        f.write("Target cell line:  {}\n".format(args.target_cell_line))
        f.write("Target assay:      {}\n".format(args.target_assay))
        f.write("Tracks:            {}\n".format(track_desc))
        f.write("\nRESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("Pairs evaluated:        {}\n".format(len(df_results)))
        f.write("Pearson r  (pred_delta):  {:.4f}  (p={:.2e})\n".format(pearson_r, pearson_p))
        f.write("Spearman r (pred_delta):  {:.4f}  (p={:.2e})\n".format(spearman_r, spearman_p))
        f.write("Pearson r  (log2 softplus):  {:.4f}  (p={:.2e})\n".format(log_pear_r, log_pear_p))
        f.write("Spearman r (log2 softplus):  {:.4f}  (p={:.2e})\n".format(log_spr_r, log_spr_p))
        f.write("=" * 80 + "\n")

    print("Summary saved to: {}".format(summary_path))
    return df_results


if __name__ == "__main__":
    main()
