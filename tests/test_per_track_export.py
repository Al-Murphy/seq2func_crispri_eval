"""``--per_track_export`` in the AlphaGenome scripts: per-track exon means must reproduce the readout."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "{}.py".format(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeModel:
    """Returns a fixed random (B, L, T) rna_seq signal, whatever the input."""

    def __init__(self, B, L, T, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.sig = torch.rand(B, L, T, generator=g) * 5

    def __call__(self, x, org_idx):
        return {"rna_seq": (None, self.sig[: x.shape[0]])}


@pytest.fixture(params=["test_fulco_alphagenome", "test_gasperini_alphagenome"])
def script(request):
    return _load(request.param)


def test_track_means_average_to_the_readout(script):
    sig = torch.rand(3, 200, 6) * 10
    exon = [5, 6, 7, 50, 51, 120]
    pt = script._exon_track_means(sig, exon)
    assert pt.shape == (3, 6)
    ref = script._aggregate_sig(sig, "rna_seq", 640, None, exon)
    np.testing.assert_allclose(pt.mean(axis=1), ref, rtol=1e-6)


def test_no_exons_gives_nan(script):
    assert np.isnan(script._exon_track_means(torch.rand(2, 50, 4), [])).all()


def test_hook_leaves_the_prediction_unchanged(script):
    B, L, T = 4, 300, 12
    model = _FakeModel(B, L, T)
    bundle = {"plus": [0, 1, 2, 3, 8, 9, 10, 11], "minus": [4, 5, 6, 7, 8, 9, 10, 11]}
    exon = list(range(100, 160))
    x = torch.zeros(B, L, 4)
    args = (model, x, 0, bundle, 640, "cpu", "rna_seq", None, exon, False)
    base = script._predict_one_shift(*args)
    out = []
    hooked = script._predict_one_shift(*args, per_track_out=out)
    assert torch.equal(base, hooked)
    assert len(out) == 1 and out[0].shape == (B, len(bundle["plus"]))
    np.testing.assert_allclose(out[0].mean(axis=1), base.numpy(), rtol=1e-6)


def test_npz_loads_without_pickle(script, tmp_path):
    import argparse
    import pandas as pd

    class M:  # API-style model: carries the condition's track metadata
        track_metadata = pd.DataFrame({"name": ["a", "b", "c"], "strand": ["+", "-", "."]})

    args = argparse.Namespace(save_path=str(tmp_path), num_shards=8, shard_idx=2)
    df = pd.DataFrame({"gene_id": ["ENSG1", "ENSG2"], "enh_loc": ["chr1:1-2", "chr1:3-4"]})
    pt_wt = [np.ones(2), np.ones(2)]
    pt_cr = [np.zeros((5, 2)), np.zeros((5, 2))]
    script._save_per_track(args, M(), {"plus": [0, 2], "minus": [1, 2]}, "P", df, pt_wt, pt_cr)
    z = np.load(tmp_path / "P_shard002of008_per_track.npz")  # allow_pickle=False by default
    assert list(z["gene_id"]) == ["ENSG1", "ENSG2"]
    assert list(z["track_name"]) == ["a", "c"] and list(z["track_strand"]) == ["+", "."]
    assert list(z["track_imputation_mode"]) == ["observed", "observed"]  # no column -> measured
    assert z["wt"].shape == (2, 2) and z["crispri"].shape == (2, 5, 2)


def test_subset_formula_reproduces_pred_delta(script):
    """The docstring formula on all tracks == the script's pred_delta on the pooled readout."""
    rng = np.random.default_rng(1)
    wt = rng.uniform(1, 5, (7, 8))            # pairs x tracks
    cr = rng.uniform(1, 5, (7, 50, 8))        # pairs x shuffles x tracks
    s = np.arange(8)
    sub_wt = wt[:, s].mean(1)
    sub_cr = cr[:, :, s].mean(2).mean(1)
    via_formula = (sub_wt - sub_cr) / sub_wt
    pooled_wt, pooled_cr = wt.mean(1), cr.mean(2).mean(1)
    via_script = (pooled_wt - pooled_cr) / (pooled_wt + 1e-8)
    np.testing.assert_allclose(via_formula, via_script, rtol=1e-6)
    assert "per_track_out" in script._predict_one_shift.__code__.co_varnames
