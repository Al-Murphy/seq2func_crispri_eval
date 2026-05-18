"""Smoke tests for the ``crispri_eval`` package."""


def test_top_level_imports():
    """The public API documented in the README must import cleanly."""
    from crispri_eval import (
        FulcoDataset,
        GasperiniDataset,
        fulco_results_for_correlation,
    )

    assert FulcoDataset is not None
    assert GasperiniDataset is not None
    assert callable(fulco_results_for_correlation)


def test_submodule_imports():
    """Submodules can be imported individually."""
    import crispri_eval.benchmark_utils
    import crispri_eval.dataset_utils
    import crispri_eval.datasets

    assert hasattr(crispri_eval.datasets, "FulcoDataset")
    assert hasattr(crispri_eval.datasets, "GasperiniDataset")
    assert hasattr(crispri_eval.dataset_utils, "dinucleotide_shuffle")
    assert hasattr(crispri_eval.dataset_utils, "one_hot_encode_dna")
    assert hasattr(crispri_eval.benchmark_utils, "fulco_results_for_correlation")
