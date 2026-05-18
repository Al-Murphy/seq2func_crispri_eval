# Tests

```bash
pip install -e .[dev]   # installs pytest
pytest                  # runs the full suite
pytest -k metadata      # subset by name
pytest tests/test_aggregation.py -v
```

## What is covered

| File | Layer | Scope |
|---|---|---|
| `test_package.py` | Smoke | The public API (`FulcoDataset`, `GasperiniDataset`, `fulco_results_for_correlation`) imports cleanly. |
| `test_benchmark_utils.py` | Unit | `fulco_results_for_correlation` filtering / copy semantics. |
| `test_dataset_utils.py` | Unit | `one_hot_encode_dna` (ACGT, N, empty) and `dinucleotide_shuffle` (shape, base-count preservation, reproducibility). |
| `test_metadata.py` | Integration | Every shipped file under `metadata/` loads with the expected schema and row counts. |
| `test_track_resolution.py` | Integration | Borzoi K562 RNA/CAGE and AlphaGenome K562 track lookups return the expected indices against the shipped metadata. |
| `test_aggregation.py` | Unit | TSS-bin / output-index helpers, per-script aggregation helpers (Borzoi, NTv3, AlphaGenome) on synthetic tensors, NTv3 one-hot → token mapping, exon overlap. |
| `test_scripts_cli.py` | Smoke | Every `scripts/*.py` responds to `--help` without raising and defines a `__main__` guard. |

## What is NOT covered

By design, the test suite never:

- downloads pretrained model weights
- requires a GPU
- runs a full inference loop end-to-end
- exercises plotting (matplotlib backends are environment-dependent)

These cost ≥ 30 min per model on a single A100. Validate them manually with
`examples/run_all.sh` on real hardware before claiming a result is reproducible.

## Optional dependencies

Each test script imports its model package at the top (`enformer_pytorch`,
`borzoi_pytorch`, `transformers`, `alphagenome_pytorch`). If the package is
not installed, the corresponding `test_*` invocations are **skipped**, not
failed (see `_load_script_module` in `conftest.py` and the
`ModuleNotFoundError` handling in `test_scripts_cli.py`).

A clean install of `requirements.txt` makes every test runnable.

## Timing

Most tests run in <1 s. The exceptions are the CLI `--help` checks, which
spawn a fresh Python interpreter per script — slow on NFS-backed installs
(the import of `enformer_pytorch` alone takes ~60 s the first time). The
hard timeout is 180 s per script; the whole suite typically finishes in
~3–5 min cold, ~30 s warm.
