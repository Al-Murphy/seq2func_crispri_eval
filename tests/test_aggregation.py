"""
Tests for the per-script aggregation / windowing helpers.

These functions take a model output tensor and reduce it to a per-pair scalar.
They are pure tensor ops and therefore cheaply testable on synthetic inputs.
"""

import numpy as np
import pytest
import torch

from .conftest import load_script


# ---------------------------------------------------------------------------
# Borzoi: _tss_bin_from_seq_index, _aggregate_predictions
# ---------------------------------------------------------------------------

def test_borzoi_tss_bin_centre_of_window():
    """An input-sequence position at the model's central crop midpoint must map to bin 3071/3072."""
    mod = load_script("test_fulco_borzoi")
    centre_seq_idx = mod.BORZOI_SEQ_LEN // 2  # 262144
    bin_idx = mod._tss_bin_from_seq_index(centre_seq_idx, num_bins=mod.BORZOI_NUM_BINS)
    # The crop starts at 163840 bp, so 262144 - 163840 = 98304 bp into the crop;
    # at 32 bp bin width that is bin 3072.
    assert bin_idx == 3072


def test_borzoi_tss_bin_outside_crop_is_clamped():
    mod = load_script("test_fulco_borzoi")
    with pytest.warns(UserWarning, match="outside Borzoi's central crop"):
        bin_idx = mod._tss_bin_from_seq_index(0, num_bins=mod.BORZOI_NUM_BINS)
    assert bin_idx == 0


def test_borzoi_aggregate_predictions_shape_and_sum():
    """Borzoi aggregation: sum over selected tracks and central bins; one scalar per batch row."""
    mod = load_script("test_fulco_borzoi")
    # (B=2, C=7611, L=6144) but we use a smaller stand-in
    B, C, L = 2, 10, 100
    pred = torch.ones(B, C, L)
    track_inds = torch.tensor([0, 1, 2], dtype=torch.long)
    tss_bin = 50
    central_bins = 5
    out = mod._aggregate_predictions(pred, track_inds, tss_bin, central_bins)
    assert isinstance(out, list)
    assert len(out) == B
    # 3 tracks × 5 bins of 1.0 = 15 per batch row
    assert all(v == pytest.approx(15.0) for v in out)


def test_borzoi_aggregate_no_track_filter():
    mod = load_script("test_fulco_borzoi")
    pred = torch.ones(1, 4, 100)
    out = mod._aggregate_predictions(pred, None, tss_bin=50, central_bins=3)
    # 4 tracks × 3 bins
    assert out[0] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# NTv3: _tss_output_index, _aggregate
# ---------------------------------------------------------------------------

def test_ntv3_tss_output_index_at_centre():
    """An input position at the centre of the input window lands at the centre of the output crop."""
    mod = load_script("test_fulco_ntv3")
    centre = mod.NTV3_SEQ_LEN // 2
    idx, pred_len = mod._tss_output_index(centre, seq_len=mod.NTV3_SEQ_LEN)
    # The output crop is the middle 37.5%; centre of input maps to centre of output crop.
    assert idx == pred_len // 2


def test_ntv3_tss_output_index_clamped_with_warning():
    mod = load_script("test_fulco_ntv3")
    with pytest.warns(UserWarning, match="outside NTv3"):
        idx, pred_len = mod._tss_output_index(0, seq_len=mod.NTV3_SEQ_LEN)
    assert idx == 0
    assert pred_len == int(mod.NTV3_SEQ_LEN * mod.NTV3_PRED_CENTER_FRACTION)


def test_ntv3_aggregate_sums_tracks_and_window():
    mod = load_script("test_fulco_ntv3")
    pred = torch.ones(2, 1000, 10)  # (B, L_pred, T)
    track_inds = torch.tensor([0, 1], dtype=torch.long)
    out = mod._aggregate(pred, track_inds, tss_output_idx=500, window_bp=20, apply_softplus=False)
    # 2 tracks × 20 bp of 1.0
    assert len(out) == 2
    assert all(v == pytest.approx(40.0) for v in out)


def test_ntv3_aggregate_applies_softplus():
    mod = load_script("test_fulco_ntv3")
    pred = -10.0 * torch.ones(1, 100, 1)  # very negative
    track_inds = torch.tensor([0], dtype=torch.long)
    no_sp = mod._aggregate(pred, track_inds, tss_output_idx=50, window_bp=4, apply_softplus=False)
    with_sp = mod._aggregate(pred, track_inds, tss_output_idx=50, window_bp=4, apply_softplus=True)
    # softplus(-10) ≈ 4.5e-5 per position; raw is -10 per position
    assert no_sp[0] < 0
    assert with_sp[0] > 0
    assert with_sp[0] == pytest.approx(0.0, abs=1e-3)


def test_ntv3_onehot_to_tokens_acgt_mapping():
    mod = load_script("test_fulco_ntv3")
    # one-hot ACGT (channel order ACGT) on CPU
    eye = torch.eye(4)
    x = eye.unsqueeze(0)  # (1, 4, 4)
    tokens = mod.onehot_to_ntv3_tokens(x, device=torch.device("cpu"))
    # NTV3 vocab: A=6, C=8, G=9, T=7
    assert tokens.shape == (1, 4)
    assert tokens[0].tolist() == [6, 8, 9, 7]


def test_ntv3_onehot_n_position_becomes_n_token():
    mod = load_script("test_fulco_ntv3")
    # All-zeros row should become the N token (10)
    x = torch.zeros(1, 3, 4)
    x[0, 0] = torch.tensor([1, 0, 0, 0])  # A
    x[0, 2] = torch.tensor([0, 0, 0, 1])  # T
    tokens = mod.onehot_to_ntv3_tokens(x, device=torch.device("cpu"))
    assert tokens[0].tolist() == [6, 10, 7]  # A, N, T


# ---------------------------------------------------------------------------
# AlphaGenome: _aggregate_sig, _exon_seq_indices
# ---------------------------------------------------------------------------

def test_alphagenome_aggregate_sig_window():
    mod = load_script("test_fulco_alphagenome")
    sig = torch.ones(3, 100, 5)
    # No exon mask → sum over a TSS-centred window of selected tracks.
    # window_bp >= 100 centred at 50 covers all 100 positions × 5 tracks = 500.
    out = mod._aggregate_sig(sig, "rna_seq", window_bp=200, center_index=50, exon_idx=None)
    assert len(out) == 3
    assert all(v == pytest.approx(500.0) for v in out)


def test_alphagenome_aggregate_sig_exon_mean():
    mod = load_script("test_fulco_alphagenome")
    # value == position index over 5 tracks; exon-mean of positions 10..14 = 12
    pos = torch.arange(100, dtype=torch.float32).view(1, 100, 1).expand(1, 100, 5).clone()
    out = mod._aggregate_sig(pos, "rna_seq", window_bp=0, center_index=None,
                             exon_idx=[10, 11, 12, 13, 14])
    assert out[0] == pytest.approx(12.0)


def test_alphagenome_exon_indices_overlap_plus_strand():
    mod = load_script("test_fulco_alphagenome")
    # One exon entirely inside the window
    exon_coords = {"ENSG0001": [("chr1", 1100, 1200)]}
    out = mod._exon_seq_indices(
        gene_id="ENSG0001", exon_coords=exon_coords, chrom="chr1",
        seq_start=1000, seq_len=500, strand=1,
    )
    assert out == list(range(100, 200))


def test_alphagenome_exon_indices_chr_prefix_tolerant():
    mod = load_script("test_fulco_alphagenome")
    exon_coords = {"ENSG0001": [("1", 1100, 1200)]}  # no chr prefix
    out = mod._exon_seq_indices("ENSG0001", exon_coords, "chr1", 1000, 500, strand=1)
    assert out == list(range(100, 200))


def test_alphagenome_exon_indices_reverse_complement_flip():
    mod = load_script("test_fulco_alphagenome")
    exon_coords = {"G": [("chr1", 1100, 1110)]}
    fwd = mod._exon_seq_indices("G", exon_coords, "chr1", 1000, 500, strand=1)
    rev = mod._exon_seq_indices("G", exon_coords, "chr1", 1000, 500, strand=-1)
    # Reverse complement: each position p -> (seq_len - 1 - p). The function
    # does NOT re-sort after the flip, so positions come back in descending order.
    assert fwd == list(range(100, 110))
    assert rev == [500 - 1 - p for p in fwd]
    # Sanity-check: the underlying *set* of positions is what matters for
    # downstream aggregation (mean over positions is order-invariant).
    assert sorted(rev) == sorted(500 - 1 - p for p in fwd)


def test_alphagenome_exon_indices_no_overlap():
    mod = load_script("test_fulco_alphagenome")
    exon_coords = {"G": [("chr2", 100, 200)]}   # wrong chromosome
    assert mod._exon_seq_indices("G", exon_coords, "chr1", 0, 500, strand=1) == []

    exon_coords = {"G": [("chr1", 5000, 6000)]}  # off-window
    assert mod._exon_seq_indices("G", exon_coords, "chr1", 0, 500, strand=1) == []


def test_alphagenome_exon_indices_missing_gene():
    mod = load_script("test_fulco_alphagenome")
    assert mod._exon_seq_indices("MISSING", {}, "chr1", 0, 500, strand=1) == []


# ---------------------------------------------------------------------------
# Enformer: _enformer_predict mock
# ---------------------------------------------------------------------------

def test_enformer_predict_picks_human_head():
    """Sanity-check that the wrapper picks the ``human`` key from Enformer's output dict."""
    mod = load_script("test_fulco_enformer")

    class FakeModel:
        def __call__(self, x):
            return {"human": torch.full((1, 4, 5), 7.0), "mouse": torch.zeros(1, 4, 5)}

    out = mod._enformer_predict(FakeModel(), torch.zeros(1, 4, 4))
    assert out.shape == (1, 4, 5)
    assert torch.all(out == 7.0)
