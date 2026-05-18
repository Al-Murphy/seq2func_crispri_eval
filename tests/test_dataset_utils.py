"""Unit tests for ``crispri_eval.dataset_utils``."""

import numpy as np
import pytest
import torch

from crispri_eval.dataset_utils import dinucleotide_shuffle, one_hot_encode_dna


# ---------------------------------------------------------------------------
# one_hot_encode_dna
# ---------------------------------------------------------------------------

def test_one_hot_acgt_is_identity():
    """ACGT must map to the canonical identity ordering."""
    out = one_hot_encode_dna("ACGT")
    expected = np.eye(4)
    np.testing.assert_array_equal(out, expected)


def test_one_hot_lowercase_uppercased():
    np.testing.assert_array_equal(
        one_hot_encode_dna("acgt"),
        one_hot_encode_dna("ACGT"),
    )


def test_one_hot_n_is_zero_row():
    out = one_hot_encode_dna("ANC")
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[1], [0, 0, 0, 0])
    np.testing.assert_array_equal(out[0], [1, 0, 0, 0])
    np.testing.assert_array_equal(out[2], [0, 1, 0, 0])


def test_one_hot_empty_sequence():
    out = one_hot_encode_dna("")
    assert out.shape == (0, 4)


def test_one_hot_single_base_returns_1d():
    """The function returns a 1-D vector for length-1 sequences (existing behaviour)."""
    out = one_hot_encode_dna("A")
    assert out.shape == (4,)
    np.testing.assert_array_equal(out, [1, 0, 0, 0])


def test_one_hot_each_row_is_one_hot_or_zero():
    """Every row must sum to 1 (canonical) or 0 (N/unknown)."""
    seq = "ACGTACGTACGTNNNN"
    out = one_hot_encode_dna(seq)
    sums = out.sum(axis=1)
    assert set(sums.tolist()) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# dinucleotide_shuffle
# ---------------------------------------------------------------------------

def _make_one_hot_seq(s: str) -> torch.Tensor:
    """Helper: (1, 4, L) tensor of one-hot DNA in NCL format."""
    arr = one_hot_encode_dna(s)        # (L, 4)
    t = torch.from_numpy(arr).float()  # (L, 4)
    return t.T.unsqueeze(0)            # (1, 4, L)


# Use a sequence with diverse dinucleotide composition. Pure repeats like
# "ACGTACGT..." trip the algorithm's "no diversity" guard.
_DIVERSE_SEQ_64 = (
    "AACTGCATGCTAGCTAGCTAGCATCGATCGTAGCTAGCTAGAACTGCATGCTAGCTAGCTAGCA"
)


def test_dinucleotide_shuffle_shape():
    x = _make_one_hot_seq(_DIVERSE_SEQ_64[:32])
    out = dinucleotide_shuffle(x, n=5, random_state=0)
    # Documented output shape: (-1, n, k, -1) — (batch, n_shuffles, alphabet, length)
    assert out.shape == (1, 5, 4, 32)


def test_dinucleotide_shuffle_preserves_base_counts():
    """A dinucleotide shuffle preserves the per-base composition exactly."""
    x = _make_one_hot_seq(_DIVERSE_SEQ_64)
    out = dinucleotide_shuffle(x, n=3, random_state=42)

    orig_counts = x[0].sum(dim=-1)  # (4,)
    for i in range(out.shape[1]):
        shuf_counts = out[0, i].sum(dim=-1)
        torch.testing.assert_close(shuf_counts, orig_counts)


def test_dinucleotide_shuffle_is_reproducible_with_seed():
    x = _make_one_hot_seq(_DIVERSE_SEQ_64)
    a = dinucleotide_shuffle(x, n=3, random_state=123)
    b = dinucleotide_shuffle(x, n=3, random_state=123)
    torch.testing.assert_close(a, b)


def test_dinucleotide_shuffle_actually_shuffles():
    """For a non-trivial sequence, at least one shuffle should differ from the original."""
    x = _make_one_hot_seq(_DIVERSE_SEQ_64)
    out = dinucleotide_shuffle(x, n=10, random_state=0)
    any_changed = any(not torch.equal(out[0, i], x[0]) for i in range(out.shape[1]))
    assert any_changed, "dinucleotide_shuffle returned the original sequence for every n"
