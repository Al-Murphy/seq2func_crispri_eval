"""
Shared pytest fixtures and path helpers.

The test scripts live in ``scripts/`` and import from the ``crispri_eval``
package. We add the project root to ``sys.path`` so scripts can be imported
via ``importlib`` without a separate install step.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
METADATA_DIR = REPO_ROOT / "metadata"

# Ensure ``crispri_eval`` is importable when tests are run without `pip install -e .`
sys.path.insert(0, str(REPO_ROOT))


def _load_script_module(script_path: Path):
    """Import a top-level script as a Python module without running ``main()``.

    If the script's model-specific imports are not installed in the current
    environment (e.g. ``borzoi_pytorch`` for the Borzoi scripts), the test
    that loaded it is skipped rather than failed.
    """
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        pytest.skip(
            "Skipping {}: optional dependency missing ({}).".format(
                script_path.name, exc.name
            )
        )
    return module


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def scripts_dir() -> Path:
    return SCRIPTS_DIR


@pytest.fixture(scope="session")
def metadata_dir() -> Path:
    return METADATA_DIR


@pytest.fixture(scope="session")
def all_test_scripts():
    return sorted(SCRIPTS_DIR.glob("test_*.py"))


@pytest.fixture(scope="session")
def all_plot_scripts():
    return sorted(SCRIPTS_DIR.glob("plot_*.py"))


def load_script(name: str):
    """Convenience: ``load_script('test_fulco_borzoi')`` returns the imported module."""
    return _load_script_module(SCRIPTS_DIR / f"{name}.py")
