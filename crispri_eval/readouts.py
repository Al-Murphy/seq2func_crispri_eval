"""
Gene-level RNA-Seq readouts for the 32 bp Calico models (Borzoi / Flashzoi, Cerberus).

The ``exon`` readout scores a gene the way AlphaGenome's RNA-Seq readout does in
``test_*_alphagenome.py``, and the way Calico scores genes (Borzoi / Cerberus papers,
baskerville ``snps.py`` covgene):

  * only the output bins that overlap the gene's exons (GENCODE, via the shipped
    ``metadata/*_gene_exons_hg38.csv`` tables) are read, wherever they fall in the
    window, instead of a fixed window on the TSS;
  * only strand-matched tracks are used. The benchmark reverse-complements minus-strand
    genes, so in input orientation every gene is on '+': stranded tracks whose
    identifier ends in '-' are antisense and are dropped; '+' and unstranded tracks
    are kept;
  * predictions are converted back to coverage units with Calico's own untransform
    before summing (baskerville ``dataset.py`` ``untransform_preds_borzoi`` /
    ``untransform_preds``, ``unscale=False``).

Scoring is unchanged: the fractional drop of the summed signal, computed after averaging
replicates, forward strand only.
"""

import numpy as np
import pandas as pd
import torch


def load_exon_table(csv_path):
    """``{gene_id: [(chrom, start, end), ...]}`` from a [gene_id, chrom, start, end] CSV.

    Coordinates are 0-based half-open (the AlphaGenome scripts' cache format).
    """
    df = pd.read_csv(csv_path)
    missing = {"gene_id", "chrom", "start", "end"}.difference(df.columns)
    if missing:
        raise KeyError("{} lacks column(s) {}".format(csv_path, sorted(missing)))
    return {
        gid: list(zip(g["chrom"].astype(str), g["start"].astype(int), g["end"].astype(int)))
        for gid, g in df.groupby("gene_id")
    }


def exon_seq_indices(gene_id, exon_coords, chrom, seq_start, seq_len, strand):
    """Exon positions of ``gene_id`` as 0-based indices into the model input.

    Same mapping as ``_exon_seq_indices`` in ``test_*_alphagenome.py``: positions are
    clipped to the window and mirrored for minus-strand genes, whose input is
    reverse-complemented by the dataset.
    """
    chrom_str = str(chrom).replace("chr", "", 1)
    seq_end = seq_start + seq_len
    parts = []
    for ex_chrom, ex_start, ex_end in exon_coords.get(gene_id, []):
        if str(ex_chrom).replace("chr", "", 1) != chrom_str:
            continue
        lo = max(ex_start, seq_start) - seq_start
        hi = min(ex_end, seq_end) - seq_start
        if lo < hi:
            parts.append(np.arange(lo, hi, dtype=np.int64))
    if not parts:
        return np.zeros(0, dtype=np.int64)
    pos = np.unique(np.concatenate(parts))
    if strand < 1:
        pos = seq_len - 1 - pos
    return pos


def exon_bins(seq_indices, crop_bp, bin_bp, num_bins):
    """Output bins (sorted, unique) overlapping the given input positions.

    ``crop_bp`` is the number of input bp before the first output bin (163,840 for
    Borzoi, 0 for Cerberus). Positions outside the output are dropped.
    """
    pos = np.asarray(seq_indices, dtype=np.int64) - int(crop_bp)
    pos = pos[(pos >= 0) & (pos < num_bins * bin_bp)]
    return np.unique(pos // bin_bp)


def gene_exon_bins(item, exon_coords, seq_len, crop_bp, bin_bp, num_bins):
    """Exon bins for one dataset item (window = TSS - seq_len // 2, as the datasets build it)."""
    chrom = str(item["enh_loc"]).split(":")[0]
    seq_start = int(item["tss"]) - seq_len // 2
    idx = exon_seq_indices(item["gene_id"], exon_coords, chrom, seq_start, seq_len,
                           int(item["strand"]))
    return exon_bins(idx, crop_bp, bin_bp, num_bins)


def pair_readout_bins(readout, item, exon_coords, tss_window, seq_len, crop_bp, bin_bp, num_bins):
    """(bins, readout used) for one pair. ``exon`` falls back to ``tss_window`` when the
    gene has no exon bin inside the output (absent from the table, or out of range)."""
    if readout == "exon":
        bins = gene_exon_bins(item, exon_coords, seq_len, crop_bp, bin_bp, num_bins)
        if len(bins):
            return bins, "exon"
    return np.asarray(tss_window, dtype=np.int64), "tss"


def strand_matched_mask(identifiers):
    """True for tracks read on the gene's strand in input orientation ('+' or unstranded)."""
    return np.array([not str(s).endswith("-") for s in identifiers], dtype=bool)


def borzoi_exon_setup(target_csv, track_inds, device):
    """Strand-matched subset of ``track_inds`` and an untransform closure for it.

    ``target_csv`` is Borzoi's targets_human.txt (first column = head index); the
    untransform is baskerville ``untransform_preds_borzoi`` with ``unscale=False``.
    """
    tmeta = pd.read_csv(target_csv, sep="\t", index_col=0).loc[list(track_inds)]
    keep = strand_matched_mask(tmeta["identifier"])
    tmeta = tmeta[keep]
    kept = [int(t) for t, k in zip(track_inds, keep) if k]

    def _t(v):
        return torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)

    scale = _t(tmeta["scale"].values)
    clip_soft = _t(tmeta["clip_soft"].values if "clip_soft" in tmeta
                   else np.full(len(tmeta), np.inf))
    sqrt_mask = torch.as_tensor(tmeta["sum_stat"].astype(str).str.contains("_sqrt").values,
                                device=device)

    def untransform(y):
        return untransform_borzoi(y, scale, clip_soft, sqrt_mask)

    return kept, untransform


def cerberus_exon_setup(target_csv, track_inds, device):
    """(strand-matched mask over ``track_inds``, untransform closure for those tracks).

    ``target_csv`` is Cerberus's targets_human.txt (first column = head index); the
    untransform is baskerville ``untransform_preds`` with ``unscale=False``. Tracks are
    not dropped here: the Cerberus scripts keep every track in their per-track output
    and apply the mask when they sum a track set.
    """
    tmeta = pd.read_csv(target_csv, sep="\t", index_col=0).loc[list(track_inds)]
    sm = strand_matched_mask(tmeta["identifier"])
    sqrt_mask = torch.as_tensor(tmeta["sum_stat"].astype(str).str.contains("_sqrt").values,
                                device=device)
    clip_soft = None
    if "clip_soft" in tmeta:
        clip_soft = torch.as_tensor(tmeta["clip_soft"].values.astype(np.float32), device=device)

    def untransform(y):
        return untransform_cerberus(y, sqrt_mask, clip_soft)

    return sm, untransform


def untransform_borzoi(y, scale, clip_soft, sqrt_mask):
    """baskerville ``untransform_preds_borzoi`` (``unscale=False``) on (B, T, bins) tensors.

    ``scale``, ``clip_soft``: (T,) tensors; ``sqrt_mask``: (T,) bool tensor.
    """
    s = scale.view(1, -1, 1)
    cs = clip_soft.view(1, -1, 1)
    y = y / s
    y = torch.where(y > cs, cs + (y - cs) ** 2, y)
    y = torch.where(sqrt_mask.view(1, -1, 1), y.clamp(min=0) ** (4.0 / 3.0), y)
    return y * s


def untransform_cerberus(y, sqrt_mask, clip_soft=None):
    """baskerville ``untransform_preds`` (``unscale=False``) on (B, T, bins) tensors.

    ``sqrt_mask``: (T,) bool tensor; ``clip_soft``: optional (T,) tensor (the Cerberus
    K562 RNA tracks have none).
    """
    if clip_soft is not None:
        cs = clip_soft.view(1, -1, 1)
        y = torch.where(y > cs, cs - 1 + (y - cs + 1) ** 2, y)
    return torch.where(sqrt_mask.view(1, -1, 1), -1 + (y + 1) ** 2, y)
