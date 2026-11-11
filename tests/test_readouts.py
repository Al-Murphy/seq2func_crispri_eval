"""Unit tests for ``crispri_eval.readouts`` (exon readout for Borzoi / Cerberus)."""

import numpy as np
import pandas as pd
import pytest
import torch

from crispri_eval import readouts

from .conftest import METADATA_DIR, load_script


def test_exon_seq_indices_plus_and_minus_strand():
    coords = {"G": [("1", 100, 104), ("2", 0, 50)]}  # second exon on another chrom
    plus = readouts.exon_seq_indices("G", coords, "chr1", seq_start=90, seq_len=40, strand=1)
    assert plus.tolist() == [10, 11, 12, 13]
    # Minus-strand inputs are reverse-complemented: positions mirror about the window.
    minus = readouts.exon_seq_indices("G", coords, "chr1", seq_start=90, seq_len=40, strand=-1)
    assert sorted(minus.tolist()) == [26, 27, 28, 29]


def test_exon_seq_indices_clips_to_window_and_handles_missing_gene():
    coords = {"G": [("X", 0, 1000)]}
    idx = readouts.exon_seq_indices("G", coords, "chrX", seq_start=990, seq_len=20, strand=1)
    assert idx.tolist() == list(range(10))
    assert len(readouts.exon_seq_indices("OTHER", coords, "chrX", 0, 20, 1)) == 0


def test_exon_bins_respects_crop():
    # Borzoi-like geometry: 100 bp crop each side, 10 output bins of 32 bp.
    pos = np.array([0, 99, 100, 131, 132, 419, 420])
    assert readouts.exon_bins(pos, crop_bp=100, bin_bp=32, num_bins=10).tolist() == [0, 1, 9]


def test_strand_matched_mask():
    m = readouts.strand_matched_mask(["ENCFF1+", "ENCFF1-", "ENCFF2", "recount3_x"])
    assert m.tolist() == [True, False, True, True]


def test_pair_readout_bins_falls_back_to_tss_window():
    item = {"enh_loc": "chr1:5-10", "tss": 1000, "gene_id": "MISSING", "strand": 1}
    bins, used = readouts.pair_readout_bins("exon", item, {}, [7, 8, 9], 2048, 0, 32, 64)
    assert used == "tss" and bins.tolist() == [7, 8, 9]
    item["gene_id"] = "G"
    coords = {"G": [("1", 1000, 1064)]}  # 64 bp at the TSS -> input 1024..1087
    bins, used = readouts.pair_readout_bins("exon", item, coords, [7], 2048, 0, 32, 64)
    assert used == "exon" and bins.tolist() == [32, 33]


def _borzoi_transform(x, scale, clip_soft):
    """Borzoi's forward target transform (x ** 0.75, soft clip, scale)."""
    y = x ** 0.75
    y = np.where(y > clip_soft, clip_soft + np.sqrt(y - clip_soft), y)
    return y * scale


def test_untransform_borzoi_inverts_transform():
    x = np.array([0.0, 1.0, 50.0, 3000.0, 20000.0])  # last ones exceed clip_soft
    scale, cs = 0.3, 384.0
    y = torch.tensor(_borzoi_transform(x, scale, cs), dtype=torch.float64).view(1, 1, -1)
    back = readouts.untransform_borzoi(
        y, torch.tensor([scale], dtype=torch.float64), torch.tensor([cs], dtype=torch.float64),
        torch.tensor([True]),
    )
    np.testing.assert_allclose(back.view(-1).numpy() / scale, x, rtol=1e-9, atol=1e-9)


def test_untransform_cerberus_inverts_sqrt():
    x = np.array([0.0, 0.5, 10.0, 4000.0])
    y = torch.tensor(np.sqrt(x + 1) - 1, dtype=torch.float64).view(1, 1, -1)
    back = readouts.untransform_cerberus(y, torch.tensor([True]))
    np.testing.assert_allclose(back.view(-1).numpy(), x, rtol=1e-9, atol=1e-9)
    # Non-sqrt tracks pass through unchanged.
    same = readouts.untransform_cerberus(y, torch.tensor([False]))
    assert torch.equal(same, y)


def test_borzoi_exon_setup_keeps_strand_matched_k562_rna():
    target_csv = METADATA_DIR / "borzoi_human_targets.txt"
    if not target_csv.exists():
        pytest.skip("borzoi_human_targets.txt not shipped")
    df = pd.read_csv(target_csv, sep="\t", index_col=0)
    d = df["description"].astype(str)
    k562 = df.index[d.str.startswith("RNA:") & d.str.contains("K562")].tolist()
    kept, untransform = readouts.borzoi_exon_setup(target_csv, k562, "cpu")
    ids = df.loc[kept, "identifier"].astype(str)
    assert 0 < len(kept) < len(k562)
    assert not ids.str.endswith("-").any()
    out = untransform(torch.zeros(1, len(kept), 3))
    assert out.shape == (1, len(kept), 3) and torch.all(out == 0)


def test_borzoi_aggregate_bins_matches_tss_window():
    mod = load_script("test_fulco_borzoi")
    pred = torch.rand(2, 6, 200)
    tracks = torch.tensor([0, 2, 5])
    bins = torch.as_tensor(mod._tss_window_bins(100, 20, 200))
    a = mod._aggregate_bins(pred, tracks, bins)
    b = mod._aggregate_predictions(pred, tracks, 100, 20)
    assert a == pytest.approx(b)
    assert len(bins) == 21
