"""Per-pair seeding of the enhancer shuffles (``crispri_eval.datasets``)."""

import numpy as np
import pytest
import torch

from crispri_eval.datasets import FulcoDataset, GasperiniDataset, pair_shuffle_seed

ENH_MID, ENH_BP = 5_000, 2_000


def _genome(L=10_000, seed=0):
    idx = np.random.default_rng(seed).integers(0, 4, L)
    x = torch.zeros(1, L, 4)
    x[0, torch.arange(L), torch.as_tensor(idx)] = 1.0
    return x


def _dataset(cls, seed=101, N=5):
    ds = object.__new__(cls)  # skip __init__: _crispri needs only these attributes
    ds.N, ds.seed, ds.crispri_perturb_mode = N, seed, "dinucleotide"
    return ds


def _insert(out, win_start):
    a = ENH_MID - ENH_BP // 2 - win_start
    return out[:, a:a + ENH_BP]


def test_pair_seed_is_deterministic_and_in_range():
    s = pair_shuffle_seed(101, "ENSG1", "chr1:1-2")
    assert s == pair_shuffle_seed(101, "ENSG1", "chr1:1-2")
    assert 0 <= s < 10_000_000


def test_pair_seed_none_means_unseeded():
    assert pair_shuffle_seed(None, "ENSG1", "chr1:1-2") is None


@pytest.mark.parametrize("other", [
    (102, "ENSG1", "chr1:1-2", 0),
    (101, "ENSG2", "chr1:1-2", 0),
    (101, "ENSG1", "chr1:1-3", 0),
    (101, "ENSG1", "chr1:1-2", 43),
])
def test_pair_seed_depends_on_every_key(other):
    assert pair_shuffle_seed(101, "ENSG1", "chr1:1-2", 0) != pair_shuffle_seed(*other)


@pytest.mark.parametrize("cls", [FulcoDataset, GasperiniDataset])
def test_same_shuffles_whatever_the_window(cls):
    """Two models with different windows over the same enhancer get the same shuffled bases."""
    g, ds = _genome(), _dataset(cls)
    s = pair_shuffle_seed(101, "ENSG1", "chr1:4000-6000")
    small = ds._crispri(g[:, 2_000:8_000].clone(), ENH_MID, 2_000, ENH_BP, shuffle_seed=s)
    large = ds._crispri(g[:, 1_000:9_000].clone(), ENH_MID, 1_000, ENH_BP, shuffle_seed=s)
    assert small.shape == (ds.N, 6_000, 4) and large.shape == (ds.N, 8_000, 4)
    torch.testing.assert_close(_insert(small, 2_000), _insert(large, 1_000), rtol=0, atol=0)
    # flanks are untouched, and the insert really was shuffled
    torch.testing.assert_close(small[:, :2_000], g[:, 2_000:4_000].expand(ds.N, -1, -1))
    assert not torch.equal(_insert(small, 2_000)[0], g[0, 4_000:6_000])


@pytest.mark.parametrize("cls", [FulcoDataset, GasperiniDataset])
def test_seed_none_redraws(cls):
    g, ds = _genome(), _dataset(cls, seed=None)
    a = ds._crispri(g[:, 2_000:8_000].clone(), ENH_MID, 2_000, ENH_BP, shuffle_seed=None)
    b = ds._crispri(g[:, 2_000:8_000].clone(), ENH_MID, 2_000, ENH_BP, shuffle_seed=None)
    assert not torch.equal(a, b)
