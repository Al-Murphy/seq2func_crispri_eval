"""
crispri_eval
============

Zero-shot evaluation of sequence-to-function genomic deep-learning models on
CRISPRi enhancer-knockdown screens (Fulco et al. 2019, Gasperini et al. 2019).

Dataset loaders centre a model's receptive field on each gene's TSS, score
the wild-type window, then dinucleotide-shuffle the enhancer slice and score
the perturbed window. The predicted fold-change in the central output bin(s)
is correlated with the measured CRISPRi effect.

Modules
-------
datasets         FulcoDataset, GasperiniDataset
dataset_utils    Genome fetcher, one-hot encoding, dinucleotide shuffle
benchmark_utils  Row-selection helpers for correlation analysis
"""

from .datasets import FulcoDataset, GasperiniDataset
from .benchmark_utils import fulco_results_for_correlation

__all__ = [
    "FulcoDataset",
    "GasperiniDataset",
    "fulco_results_for_correlation",
]
