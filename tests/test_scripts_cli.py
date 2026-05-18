"""
CLI smoke tests: every test_*.py and plot_*.py script must respond to ``--help``
without raising. This catches argparse misconfiguration (mismatched defaults,
typo'd choices) and circular import bugs without needing a model or GPU.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def _all_scripts():
    """Discover the scripts at collection time so pytest parametrizes them."""
    here = Path(__file__).resolve().parent.parent / "scripts"
    return sorted(here.glob("*.py"))


@pytest.mark.parametrize("script", _all_scripts(), ids=lambda p: p.name)
def test_script_help_exits_zero(script, repo_root):
    """``python scripts/<name>.py --help`` must exit 0.

    If the script imports an optional model package that is not installed
    (``borzoi_pytorch``, ``enformer_pytorch``, ...), the test is skipped
    rather than failed — those imports are guaranteed to fail before
    argparse runs.
    """
    # 180 s timeout: enformer_pytorch and friends import slowly on NFS-backed
    # storage (~60-90 s cold), and we want the test to stay green there.
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 and "ModuleNotFoundError" in result.stderr:
        pytest.skip(
            "Skipping {}: optional dependency missing\n{}".format(
                script.name, result.stderr.strip().splitlines()[-1]
            )
        )
    assert result.returncode == 0, (
        "{} --help failed.\nstdout:\n{}\nstderr:\n{}".format(
            script.name, result.stdout, result.stderr
        )
    )
    # --help output should at least mention the script's purpose
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", _all_scripts(), ids=lambda p: p.name)
def test_script_has_main_guard(script):
    """Every script must define ``if __name__ == "__main__":`` so it is importable."""
    text = script.read_text()
    assert '__name__ == "__main__"' in text or "__name__ == '__main__'" in text
