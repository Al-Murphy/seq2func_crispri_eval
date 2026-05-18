# seq2func_crispri_eval

This repo deals with current state-of-the-art sequence-to-function and pretrained 
genomic deep learning models' performance at predicting the effect across distal
__cis__-regulatory elements (CREs). 

This benchmark was first conducted by [Karollus et al., 2023](https://link.springer.com/article/10.1186/s13059-023-02899-9).


It is a zero-shot CRISPRi enhancer-perturbation evaluation for sequence-to-function
genomic deep-learning models on two pooled CRISPRi screens:

- **[Fulco et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31784727/)** (CRISPRi-FlowFISH), aligned to the Avsec / Karollus
  windows from the [SequenceModelBenchmark Zenodo](https://zenodo.org/records/8275436)
  release.
- **[Gasperini et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30612741/)** high-confidence enhancer-gene pairs.

Four models are supported out of the box:

| Model       | Backbone source                             | Context  | Output res |
|-------------|---------------------------------------------|----------|------------|
| Enformer    | HuggingFace `EleutherAI/enformer-official-rough` | 196,608 bp | 128 bp |
| Borzoi / Flashzoi | HuggingFace `johahi/borzoi-replicate-{0..3}` (`flashzoi-...`) | 524,288 bp | 32 bp |
| NTv3        | HuggingFace `InstaDeepAI/NTv3_650M_post`         | 1,048,576 bp | 1 bp |
| AlphaGenome | HuggingFace `gtca/alphagenome_pytorch` (auto-downloaded via `huggingface_hub`)        | 1,048,576 bp | 1 bp |

For each enhancer-gene pair, the model scores the wild-type sequence centred on
the gene's TSS, then scores `N` dinucleotide-shuffled negatives in which the
enhancer slice has been scrambled in place. The predicted fractional drop
`(WT - mean CRISPRi) / WT` is correlated against the measured CRISPRi effect.

> **Scope.** This repo evaluates **base / pretrained** models only. CRISPRi
> fine-tuning lives in a separate project.

---

## Repository layout

```
seq2func_crispri_eval/
├── crispri_eval/             # Python package (dataset loaders + utils)
│   ├── datasets.py           #   FulcoDataset, GasperiniDataset
│   ├── dataset_utils.py      #   genome fetch, one-hot, dinucleotide shuffle
│   └── benchmark_utils.py    #   fulco_results_for_correlation
│
├── scripts/                  # One self-contained script per (dataset × model)
│   ├── test_fulco_enformer.py        scripts/test_gasperini_enformer.py
│   ├── test_fulco_borzoi.py          scripts/test_gasperini_borzoi.py
│   ├── test_fulco_ntv3.py            scripts/test_gasperini_ntv3.py
│   ├── test_fulco_alphagenome.py     scripts/test_gasperini_alphagenome.py
│   ├── plot_fulco_results.py         scripts/plot_gasperini_results.py
│   └── …
│
├── metadata/                 # Shipped reference tables (~4 MB total)
│   ├── fulco/
│   │   ├── enhancer_knockdown_effects.tsv
│   │   └── ziga_additional_columns.tsv
│   ├── Gasperini_gene_tss_hg38_biomart_GRCh38_p14.csv
│   ├── Gasperini_gene_tss_hg19_biomart_GRCh37_p13.csv
│   ├── Gasperini_enh_hg38_ucscLiftOver.bed
│   ├── Gasperini_sign_enhancer_gene_pairs.csv
│   ├── borzoi_human_targets.txt    # Borzoi K562 RNA / CAGE indices
│   ├── track_metadata.parquet      # AlphaGenome K562 indices
│   ├── Fulco_gene_exons_hg38.csv   # AlphaGenome RNA-seq exon aggregation
│   └── Gasperini_gene_exons_hg38.csv
│
├── examples/run_all.sh       # End-to-end launch script (8 evals + 2 plots)
├── tests/                    # pytest suite (smoke + unit + metadata integrity)
├── pyproject.toml            # pytest config
├── requirements.txt
├── LICENSE                   # MIT
└── README.md                 # this file
```

Each `test_*_*.py` is **self-contained** — reading one script end-to-end shows
the full evaluation logic for that (dataset × model) pair. Shared code lives in
the `crispri_eval/` package.

---

## Installation

```bash
git clone https://github.com/<you>/seq2func_crispri_eval.git
cd seq2func_crispri_eval
python -m venv .venv && source .venv/bin/activate
pip install -e .              # installs the crispri_eval package
pip install -r requirements.txt
```

Install only the model dependencies for the models you plan to run:

```bash
pip install enformer-pytorch     # Enformer
pip install borzoi-pytorch       # Borzoi / Flashzoi
pip install transformers         # NTv3
pip install alphagenome-pytorch huggingface_hub  # AlphaGenome
```

The first run will download a reference genome to `./.cache/` via `pysam`
(~3 GB for hg38). Subsequent runs reuse it.

---

## Running one evaluation

The simplest invocation per model:

```bash
# Enformer  -- Gasperini, K562 CAGE (default), ~10 min on one A100
python scripts/test_gasperini_enformer.py

# Borzoi    -- Fulco, K562 RNA-Seq (default), ~30 min on one A100
python scripts/test_fulco_borzoi.py

# NTv3      -- Fulco, K562 RNA-Seq (ENCSR056HPM demo track), bf16 for memory
python scripts/test_fulco_ntv3.py --autocast_dtype bf16

# AlphaGenome -- Gasperini, K562 RNA-Seq with GENCODE exon aggregation
python scripts/test_gasperini_alphagenome.py
```

Each script writes:
```
<save_path>/<output_prefix>_results.csv   # one row per evaluated pair
<save_path>/<output_prefix>_summary.txt   # run config + Pearson / Spearman
```

CSV columns are consistent across models, so the plot scripts can ingest any
combination of results files.

---

## Running all four models + plotting

A reference script chains the full pipeline:

```bash
bash examples/run_all.sh
```

It will produce `./results/test_{Fulco,Gasperini}_<model>/` directories and the
combined-model plots `./results/plots/{fulco,gasperini}_all_models.png`.

---

## Plotting

```bash
python scripts/plot_fulco_results.py \
    --results ./results/test_Fulco_enformer/Fulco_base_enformer_results.csv \
              ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \
              ./results/test_Fulco_ntv3/Fulco_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
              ./results/test_Fulco_alphagenome/Fulco_AlphaGenome_base_rna_seq_results.csv \
    --labels "Enformer" "Borzoi" "NTv3" "AlphaGenome" \
    --max_width_subplots 4 \
    --observed_mode decrease_only \
    --output ./results/plots/fulco_all_models.png
```

`plot_gasperini_results.py` has the same interface (no `--observed_mode`
flag since Gasperini uses all pairs).

---

## Adding a new model

A new model needs **one script** in `scripts/test_<dataset>_<model>.py` that:

1. Loads the model from its native source (HuggingFace, local checkpoint, …).
2. Calls either `FulcoDataset` or `GasperiniDataset` from `crispri_eval`.
3. Iterates over pairs: for each pair, forward the WT and `N` shuffled
   sequences, aggregate over the model's K562 RNA / CAGE tracks at a
   TSS-centred window.
4. Computes `pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)`.
5. Writes a CSV with at minimum these columns:
   ```
   gene_id, gene_name, enh_loc, tss, tss_seq_index, strand,
   enh_dist, y_delta, pred_delta
   ```
   Optional but recommended: `pred_wt`, `pred_crispri_mean` (enables the
   log2-fold-change plots).

The four existing test scripts are intentionally parallel — diff
`test_fulco_enformer.py` against your draft to spot missing pieces. The
**only shared code** is the dataset, so each script is the ground truth for
that model's preprocessing and aggregation choices.

For shared eval-loop scaffolding (correlation, summary writing) we kept the
duplication on purpose: each script is self-contained so a reader can audit
one model's evaluation without chasing imports. If you add a fifth model and
find yourself copy-pasting more than ~40 lines of glue, consider lifting a
helper into `crispri_eval/`.

---

## Tests

```bash
pip install -e .[dev]
pytest
```

See [`tests/README.md`](tests/README.md) for what is and isn't covered. The
suite is non-GPU and uses no model weights; per-model tests skip if the
model's optional dependency (`borzoi_pytorch`, `enformer_pytorch`, …) is not
installed.

---

## Data sources

The repository ships small (~4 MB) reference tables under `metadata/`. The
larger raw data are not committed:

| Resource | Source | Used by |
|---|---|---|
| Fulco `enhancer_knockdown_effects.tsv` | Zenodo `SequenceBenchmark.zip` (Karollus et al.) → `Data/Fulco_CRISPRi/` | `FulcoDataset` (shipped) |
| Fulco `ziga_additional_columns.tsv` | same Zenodo, Karollus pipeline | `FulcoDataset` (shipped) |
| Gasperini significant pairs CSV / BED / TSS | Gasperini et al. (2019) supplement + UCSC liftover | `GasperiniDataset` (shipped) |
| Borzoi `targets_human.txt` | [calico/borzoi](https://github.com/calico/borzoi) `examples/targets_human.txt` | `*_borzoi.py` (shipped) |
| AlphaGenome `track_metadata.parquet` | [alphagenome-pytorch](https://github.com/lucidrains/alphagenome) `scripts/extract_track_metadata.py` | `*_alphagenome.py` (shipped) |
| Reference genome (hg38) | UCSC, fetched on-demand by `pysam` | all (auto-downloaded to `./.cache/`) |
| Exon CSVs (`Fulco_gene_exons_hg38.csv`, `Gasperini_gene_exons_hg38.csv`) | Ensembl REST API | `*_alphagenome.py` (shipped; regenerated on first run if absent) |
| Enformer `targets_human.txt` | [calico/basenji manuscripts/cross2020](https://github.com/calico/basenji) | `*_enformer.py` (fetched at runtime via HTTPS) |

---

## Citation

If you use this benchmark, please cite the underlying CRISPRi datasets:

- Fulco CP et al., *Nat Genet* 51 (2019), DOI 10.1038/s41588-019-0538-0
- Gasperini M et al., *Cell* 176 (2019), DOI 10.1016/j.cell.2018.11.029

…and the models you evaluate (Enformer / Borzoi / NTv3 / AlphaGenome).

## License

MIT. See [LICENSE](LICENSE).
