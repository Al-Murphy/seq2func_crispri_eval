"""
Tests for the per-model K562 track-index lookup helpers.

These are the functions that translate Borzoi / AlphaGenome track metadata
into the list of indices each test script aggregates over. If the metadata
file is reshaped or a column is renamed upstream, these tests catch it.
"""

import pytest

from .conftest import load_script


# ---------------------------------------------------------------------------
# Borzoi: get_borzoi_track_indices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def borzoi_targets_path(metadata_dir):
    return str(metadata_dir / "borzoi_human_targets.txt")


def test_borzoi_k562_rna_indices_nonempty(borzoi_targets_path):
    mod = load_script("test_fulco_borzoi")
    inds, desc = mod.get_borzoi_track_indices(borzoi_targets_path, "K562", "RNA")
    assert len(inds) > 0
    assert "K562" in desc
    assert "RNA" in desc
    # Every index must be a valid track id (0..7610)
    assert all(0 <= i < 7611 for i in inds)


def test_borzoi_k562_cage_indices_nonempty(borzoi_targets_path):
    mod = load_script("test_fulco_borzoi")
    inds, desc = mod.get_borzoi_track_indices(borzoi_targets_path, "K562", "CAGE")
    assert len(inds) > 0
    assert "CAGE" in desc


def test_borzoi_invalid_assay_raises(borzoi_targets_path):
    mod = load_script("test_fulco_borzoi")
    with pytest.raises(ValueError, match="No tracks matched"):
        mod.get_borzoi_track_indices(borzoi_targets_path, "K562", "NOT_AN_ASSAY")


def test_borzoi_missing_file_raises(tmp_path):
    mod = load_script("test_fulco_borzoi")
    with pytest.raises(FileNotFoundError):
        mod.get_borzoi_track_indices(str(tmp_path / "does_not_exist.txt"), "K562", "RNA")


def test_borzoi_gasperini_uses_same_helper(borzoi_targets_path):
    """Gasperini variant ships its own get_borzoi_track_indices — keep them in sync."""
    fulco = load_script("test_fulco_borzoi")
    gasp = load_script("test_gasperini_borzoi")
    f_inds, _ = fulco.get_borzoi_track_indices(borzoi_targets_path, "K562", "RNA")
    g_inds, _ = gasp.get_borzoi_track_indices(borzoi_targets_path, "K562", "RNA")
    assert f_inds == g_inds


# ---------------------------------------------------------------------------
# AlphaGenome: get_alphagenome_track_indices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ag_metadata_path(metadata_dir):
    return str(metadata_dir / "track_metadata.parquet")


def test_alphagenome_k562_rna_bundle(ag_metadata_path):
    mod = load_script("test_fulco_alphagenome")
    bundle, desc = mod.get_alphagenome_track_indices(
        ag_metadata_path, "K562", "rna_seq", n_total_tracks=768
    )
    assert isinstance(bundle, dict)
    for k in ("all", "plus", "minus"):
        assert k in bundle
        assert len(bundle[k]) > 0
        assert all(0 <= i < 768 for i in bundle[k])


def test_alphagenome_k562_cage_bundle(ag_metadata_path):
    mod = load_script("test_fulco_alphagenome")
    bundle, _ = mod.get_alphagenome_track_indices(
        ag_metadata_path, "K562", "cage", n_total_tracks=640
    )
    assert all(0 <= i < 640 for i in bundle["all"])


def test_alphagenome_none_metadata_returns_all_tracks():
    mod = load_script("test_fulco_alphagenome")
    bundle, desc = mod.get_alphagenome_track_indices(
        "none", "K562", "rna_seq", n_total_tracks=768
    )
    assert bundle["all"] == list(range(768))
    assert "all" in desc


def test_alphagenome_missing_metadata_raises(tmp_path):
    mod = load_script("test_fulco_alphagenome")
    with pytest.raises(FileNotFoundError):
        mod.get_alphagenome_track_indices(
            str(tmp_path / "missing.parquet"), "K562", "rna_seq", n_total_tracks=768,
        )
