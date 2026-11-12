"""Pure helpers of ``crispri_eval.alphagenome_api`` (no network, no API key)."""

import numpy as np
import torch

from crispri_eval.alphagenome_api import onehot_to_str, strand_bundle
from crispri_eval.dataset_utils import one_hot_encode_dna


def test_onehot_round_trip():
    seq = "ACGTTGCAAN"
    assert onehot_to_str(one_hot_encode_dna(seq)) == seq


def test_onehot_accepts_torch_and_padding():
    x = torch.zeros(6, 4)
    x[0, 0] = x[1, 1] = x[2, 2] = x[3, 3] = 1.0  # ACGT, then two all-zero (padding) rows
    assert onehot_to_str(x.numpy()) == "ACGTNN"


def test_strand_bundle_alphagenome2_k562():
    # Strand order of the 12 K562 RNA-seq tracks the API returns for EFO:0002067.
    strands = ["+"] * 4 + ["-"] * 4 + ["."] * 4
    b = strand_bundle(strands)
    assert b["plus"] == [0, 1, 2, 3, 8, 9, 10, 11]
    assert b["minus"] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert b["all"] == list(range(12))


def test_strand_bundle_matches_alphagenome1_rule():
    # AlphaGenome 1 K562 RNA-seq: polyA+ (+), total (+), polyA+ (-), total (-), polyA+ (.)
    b = strand_bundle(["+", "+", "-", "-", "."])
    assert len(b["plus"]) == 3 and len(b["minus"]) == 3  # the published run used 3


def test_strand_bundle_unstranded_only_falls_back_to_all():
    b = strand_bundle(["+", "+"])
    assert b["minus"] == [0, 1]  # no '-'/'.' tracks -> all, as the PyTorch selector does
    assert np.array_equal(b["plus"], [0, 1])
