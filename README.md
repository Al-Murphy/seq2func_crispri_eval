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

Six models are supported out of the box:

| Model       | Backbone source                             | Context  | Output res |
|-------------|---------------------------------------------|----------|------------|
| Enformer    | HuggingFace `EleutherAI/enformer-official-rough` | 196,608 bp | 128 bp |
| Borzoi / Flashzoi | HuggingFace `johahi/borzoi-replicate-{0..3}` (`flashzoi-...`) | 524,288 bp | 32 bp |
| Cerberus    | [calico/baskerville](https://github.com/calico/baskerville) `releases/cerberus` (8 replicates, see below) | 786,432 bp | 32 bp |
| NTv3        | HuggingFace `InstaDeepAI/NTv3_tracks_1mb` (gated, see below) | 1,048,576 bp | 1 bp |
| AlphaGenome | HuggingFace `gtca/alphagenome_pytorch` (auto-downloaded via `huggingface_hub`)        | 1,048,576 bp | 1 bp |
| AlphaGenome 2 | Hosted [AlphaGenome API](https://github.com/google-deepmind/alphagenome) (`alphagenome` client >= 2.0.0, API key, see below) | 1,048,576 bp | 1 bp |

AlphaGenome 2 (Google DeepMind, Nov 2026) is served through the AlphaGenome API
rather than released as weights. The AlphaGenome scripts take `--backend api`,
which sends each sequence to the API in place of the PyTorch port
(`crispri_eval/alphagenome_api.py`); windows, shuffles, the strand-matched track
rule and the exon readout are the same code. Each request names the K562
condition (`EFO:0002067`), for which the API returns 12 K562 RNA-Seq tracks
(ENCODE polyA+ and total RNA-Seq, paired- and single-ended, on '+', '-' and
unstranded); the strand-matched readout averages 8 of them, where AlphaGenome 1
has 5 K562 RNA-Seq tracks and averages 3. Notes:

- The hosted model is not version-pinned, so each summary records the client
  version, model version (`ALL_FOLDS`) and run date. Two runs on 2026-11-12
  were bit-identical.
- `--per_track_export` also saves every track's exon mean for the wild type and
  each shuffle (`*_per_track.npz`, shipped for the published runs), so any track
  subset can be rescored offline:
  `w = wt[:, s].mean(1); pred_delta = (w - crispri[:, :, s].mean(2).mean(1)) / w`.
  The three tracks matching AlphaGenome 1's (polyA+ and total RNA-Seq on '+',
  unstranded polyA+) score the same as all eight, so the track count does not
  explain the difference between the two models.
- Zero-shot: the AlphaGenome 2 preprint reports no CRISPRi data in training,
  and no validation metric in checkpoint selection. DeepMind did evaluate on
  the ENCODE-rE2G K562 CRISPRi benchmark, which includes both screens here.

Cerberus (Calico, Oct 2026) is Borzoi's successor: Borzoi's transformer and
U-net are replaced by bidirectional Mamba-2 (Hydra) blocks. It is released as 8
replicate models, each holding out a different fold, and every pair is scored
with all 8 (`pred_wt` and `pred_crispri_mean` are averaged across replicates
before `pred_delta`). The readout matches the Borzoi scripts (see
[Gene readouts](#gene-readouts)). Two track sets come out
of the same forward passes: `rna`, the 94 K562 RNA-Seq tracks Borzoi uses
(matched by ENCODE identifier), and `rna_allk562`, all 160 K562 RNA tracks
(adds recount3 4sU / TT-seq nascent RNA and actinomycin D shutoff tracks).

NTv3 defaults to the revised release (`NTv3_tracks_1mb`, Oct 2026), the 1 Mb
model behind the NTv3 paper's results. Its weights are gated: request access on
the HuggingFace model page (one approval covers every revised NTv3 model). The
revised model reads soft-masked DNA, so repeat-masked (lowercase) hg38 bases are
passed as lowercase tokens. The initial release (`InstaDeepAI/NTv3_650M_post`,
Dec 2025) is still supported via `--model_name InstaDeepAI/NTv3_650M_post`.

For each enhancer-gene pair, the model scores the wild-type sequence centred on
the gene's TSS, then scores `N` dinucleotide-shuffled negatives in which the
enhancer slice has been scrambled in place. The predicted fractional drop
`(WT - mean CRISPRi) / WT` is correlated against the measured CRISPRi effect.

The shuffles are seeded per pair (`--seed`, default 101), from the gene, the
enhancer and the test-time shift rather than the pair's row. Every model, shard and
window size therefore scores the same shuffled sequences, and reruns reproduce.
The results in `results/` predate the seed, except AlphaGenome 2's (seed 101),
so each of the other runs drew its own shuffles; see "Shuffles" under [Running one evaluation](#running-one-evaluation) for how much
that moves the scores.

### Gene readouts

How each model's output is reduced to one number per sequence:

| Model | Tracks | Region read | Units |
|-------|--------|-------------|-------|
| Enformer | K562 CAGE | 5 × 128 bp centred on the TSS | model output |
| Borzoi / Flashzoi | 94 K562 RNA-Seq, strand-matched (71) | output bins overlapping the gene's exons | untransformed (Calico `untransform_preds_borzoi`) |
| Cerberus | 94 (or 160) K562 RNA, strand-matched (71 / 137) | output bins overlapping the gene's exons | untransformed (Calico `untransform_preds`) |
| NTv3 | 86 K562 RNA-Seq | mean over the gene's exon bp | model output |
| AlphaGenome | 5 K562 RNA-Seq, strand-matched (3) | mean over the gene's exon bp | model output |
| AlphaGenome 2 | 12 K562 RNA-Seq, strand-matched (8) | mean over the gene's exon bp | model output |

RNA-Seq models are read over the gene's exons (GENCODE, shipped as
`metadata/*_gene_exons_hg38.csv`), which is how AlphaGenome and Calico score genes;
Enformer has no RNA-Seq output, so its CAGE tracks are read at the TSS.
"Strand-matched" keeps '+' and unstranded tracks: minus-strand genes are
reverse-complemented, so every gene sits on '+' in the model input. Borzoi and
Cerberus predict squashed coverage (a power transform plus a soft clip); their
scripts convert it back with Calico's own untransform before summing. Every
model scores the forward strand only (no reverse-complement averaging).

The Borzoi and Cerberus scripts default to `--readout exon`. `--readout tss`
reproduces the earlier readout (all tracks, transformed units, 21 × 32 bp bins
centred on the TSS); its results are kept under `*_tss_*` file names.

> **Scope.** This repo evaluates **base / pretrained** models only. CRISPRi
> fine-tuning lives in a separate project.

---

## Repository layout

```
seq2func_crispri_eval/
├── crispri_eval/             # Python package (dataset loaders + utils)
│   ├── datasets.py           #   FulcoDataset, GasperiniDataset
│   ├── dataset_utils.py      #   genome fetch, one-hot, dinucleotide shuffle
│   ├── readouts.py           #   exon readout + Calico untransforms (Borzoi, Cerberus)
│   ├── alphagenome_api.py    #   AlphaGenome 2 through the AlphaGenome API (--backend api)
│   └── benchmark_utils.py    #   fulco_results_for_correlation
│
├── scripts/                  # One self-contained script per (dataset × model)
│   ├── test_fulco_enformer.py        scripts/test_gasperini_enformer.py
│   ├── test_fulco_borzoi.py          scripts/test_gasperini_borzoi.py
│   ├── test_fulco_cerberus.py        scripts/test_gasperini_cerberus.py
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
│   ├── cerberus_human_targets.txt  # Cerberus human track list
│   ├── track_metadata.parquet      # AlphaGenome K562 indices
│   ├── Fulco_gene_exons_hg38.csv   # RNA-Seq exon readouts (all RNA-Seq models)
│   └── Gasperini_gene_exons_hg38.csv
│
├── examples/run_all.sh       # End-to-end launch script (12 evals + 2 plots)
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
pip install "alphagenome>=2.0.0"                 # AlphaGenome 2 (API client)
```

AlphaGenome 2 needs an API key from the
[AlphaGenome API](https://alphagenome.google/api). Save it in a
file only you can read; the scripts read `~/.config/alphagenome/api_key` by
default (`--api_key_file` to change it) and never print it:

```bash
mkdir -p ~/.config/alphagenome && touch ~/.config/alphagenome/api_key
chmod 600 ~/.config/alphagenome/api_key   # then paste the key into it
```

Cerberus needs its own environment: torch >=2.11,<2.14 and `mamba-ssm`
(Triton kernels), installed through calico/baskerville. Follow the baskerville
README, install this repo into that environment (`pip install -e .`), download
the weights (`cd baskerville/releases/cerberus && ./download.sh`, 4.2 GB for the
8 replicates), then point the scripts at that directory with `--cerberus_dir`
or `export CERBERUS_DIR=/path/to/baskerville/releases/cerberus`.

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

# Cerberus  -- Fulco, 94 (+160) K562 RNA-Seq tracks x 8 replicates, fp32,
#              ~3 min per pair on one H100 (shard it, see the script docstring)
python scripts/test_fulco_cerberus.py --cerberus_dir $CERBERUS_DIR

# NTv3      -- Fulco, 86 K562 RNA-Seq tracks (default), fp32
python scripts/test_fulco_ntv3.py

# AlphaGenome -- Gasperini, K562 RNA-Seq with GENCODE exon aggregation
python scripts/test_gasperini_alphagenome.py

# AlphaGenome 2 -- same script, hosted API, CPU only; 51 API calls per pair
python scripts/test_fulco_alphagenome.py --backend api --per_track_export
```

> **AlphaGenome 2 API.** Calls run in parallel threads (`--api_concurrency`,
> default 4) and retry with exponential backoff on rate limits
> (`--api_max_retries`, default 8). The published runs took ~1.4 h for Fulco
> and for each of 8 Gasperini shards (`--num_shards 8 --shard_idx i`, merged
> with `scripts/merge_tta_shards.py`), nine tasks at once; most of that was
> waiting out rate limits, so more parallel tasks will not go faster.

> **Precision.** Run NTv3 and Cerberus in fp32, the default (leave
> `--autocast_dtype` unset). The published NTv3 and Cerberus results are fp32. The score is a small difference
> between the wild-type and shuffled predictions, and eager `torch.autocast`
> bf16 introduces errors of the same size: in our tests on AlphaGenome, eager
> bf16 shifted in-silico perturbation effects by ~30% (median, relative to
> fp32) and flipped the sign of ~20% of them. We ran NTv3 and Cerberus in fp32
> on one 80 GB H100.
>
> Flashzoi needs bf16 for FlashAttention, so the published Borzoi results are
> Flashzoi in bf16. We checked the Borzoi architecture: original Borzoi
> (replicate 0, exon readout, all Fulco pairs) in bf16 vs fp32 adds a per-pair
> error of ~2.4e-3 to `pred_delta` (median |`pred_delta`| 1.7e-2), about the
> size of the noise from redrawing the 50 shuffles (~1.8e-3 per run); per-pair
> rank agreement with fp32 stays at ρ = 0.98.
>
> **Shuffles.** N = 50 is enough. We scored original Borzoi replicate 0 (fp32,
> exon readout) on Fulco with 1,000 shuffles per pair and split them into 20
> independent sets of 50. Between sets, Pearson r moved by SD 0.006 and
> Spearman ρ by SD 0.031 (range 0.27–0.38). That was the noisiest model we
> measured. From each run's recorded shuffle spread, Fulco ρ moves by SD
> 0.010–0.022 for single Flashzoi replicates, 0.005 for the 4-replicate
> ensemble and 0.014 for AlphaGenome; on Gasperini, Flashzoi ρ moves by ≤ 0.003. Fulco's
> 63 pairs alone give ρ an uncertainty of ~0.1. At N = 50 the same model's Fulco
> ρ was also ~0.015 lower than at N = 500, because a few pairs have rare shuffles
> with large effects.

Each script writes:
```
<save_path>/<output_prefix>_results.csv   # one row per evaluated pair
<save_path>/<output_prefix>_summary.txt   # run config + Pearson / Spearman
```

CSV columns are consistent across models, so the plot scripts can ingest any
combination of results files.

---

## Running all six models + plotting

A reference script chains the full pipeline:

```bash
bash examples/run_all.sh
```

It will produce `./results/test_{Fulco,Gasperini}_<model>/` directories and the
combined-model plots `./results/plots/{fulco,gasperini}_all_models.png`.
AlphaGenome 2 runs (and appears in the plots) only when an API key file is
present.

---

## Test-time augmentation (AlphaGenome RNA-Seq)

The AlphaGenome scripts support test-time augmentation (TTA) matching Karollus
et al. 2023: average the prediction over **3 sequence shifts** (`−43, 0, +43`
bp) **× forward + reverse-complement** = 6 forward passes per sequence. Each
shift re-fetches a window at `seq_start + shift` from the genome, so the
flanking bases are **real sequence** (not rolled/wrapped from the far edge);
the reverse-complement pass flips the sequence and reads the **minus-strand**
RNA-Seq tracks at flipped positions.

> Note: shuffles are drawn independently per shift rather than shared across
> shifts (as in Karollus). For the per-pair *mean* prediction this is an
> unbiased estimate of the same quantity, so it does not bias the correlation.

```bash
python scripts/test_fulco_alphagenome.py \
    --modality rna_seq --tta_shifts=-43,0,43 --tta_rev_comp
```

`--tta_shifts 0` (default) + no `--tta_rev_comp` is a single forward pass (no
TTA, original behaviour). TTA multiplies forward passes by 6, so for the 1 Mb
AlphaGenome model it is best run sharded on a cluster:

```bash
# Each array task processes pairs[shard_idx::num_shards] and writes a shard CSV.
python scripts/test_fulco_alphagenome.py \
    --modality rna_seq --tta_shifts=-43,0,43 --tta_rev_comp \
    --num_shards 8 --shard_idx $SLURM_ARRAY_TASK_ID \
    --save_path ./results/test_Fulco_alphagenome_tta/

# After all shards finish, concatenate + compute correlations:
python scripts/merge_tta_shards.py \
    --shards './results/test_Fulco_alphagenome_tta/Fulco_AlphaGenome_base_rna_seq_shard*of008_results.csv' \
    --output_prefix Fulco_AlphaGenome_base_rna_seq_tta \
    --fulco_corr_observed_subset knockdown_only
```

A ready-to-edit SLURM array template is in
[`examples/slurm_ag_tta_array.sh`](examples/slurm_ag_tta_array.sh) (set the
partition/QOS and array size for your cluster; pre-populate the `./.cache`
genome once before launching the array to avoid concurrent-download races).

---

## Plotting

```bash
python scripts/plot_fulco_results.py \
    --results ./results/test_Fulco_enformer/Fulco_base_enformer_results.csv \
              ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \
              ./results/test_Fulco_ntv3/Fulco_NTv3_InstaDeepAI_NTv3_tracks_1mb_rna_results.csv \
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
   sequences, aggregate over the model's K562 tracks: RNA-Seq over the gene's
   exons (strand-matched, in coverage units; see `crispri_eval/readouts.py`),
   CAGE over a TSS-centred window.
4. Computes `pred_delta = (pred_wt - pred_crispri_mean) / (pred_wt + 1e-8)`.
5. Writes a CSV with at minimum these columns:
   ```
   gene_id, gene_name, enh_loc, tss, tss_seq_index, strand,
   enh_dist, y_delta, pred_delta
   ```
   Optional but recommended: `pred_wt`, `pred_crispri_mean` (enables the
   log2-fold-change plots).

The existing test scripts are intentionally parallel — diff
`test_fulco_enformer.py` against your draft to spot missing pieces. The
**only shared code** is the dataset and the exon-readout helpers in
`crispri_eval/readouts.py`, so each script is the ground truth for that model's
preprocessing and aggregation choices.

For shared eval-loop scaffolding (correlation, summary writing) we kept the
duplication on purpose: each script is self-contained so a reader can audit
one model's evaluation without chasing imports. If you add another model and
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
| Cerberus `targets_human.txt` | [calico/baskerville](https://github.com/calico/baskerville) `releases/cerberus/targets_human.txt` | `*_cerberus.py` (shipped as `cerberus_human_targets.txt`) |
| AlphaGenome `track_metadata.parquet` | [alphagenome-pytorch](https://github.com/lucidrains/alphagenome) `scripts/extract_track_metadata.py` | `*_alphagenome.py` (shipped) |
| AlphaGenome 2 K562 track list | AlphaGenome API, read from a 16 kb probe prediction at the start of each run | `*_alphagenome.py --backend api` (runtime) |
| Reference genome (hg38) | UCSC, fetched on-demand by `pysam` | all (auto-downloaded to `./.cache/`) |
| Exon CSVs (`Fulco_gene_exons_hg38.csv`, `Gasperini_gene_exons_hg38.csv`) | Ensembl REST API | `*_alphagenome.py` (shipped; regenerated on first run if absent), also read by `*_ntv3.py`, `*_borzoi.py`, `*_cerberus.py` |
| Enformer `targets_human.txt` | [calico/basenji manuscripts/cross2020](https://github.com/calico/basenji) | `*_enformer.py` (fetched at runtime via HTTPS) |

---

## Citation

If you use this benchmark, please cite the underlying CRISPRi datasets:

- Fulco CP et al., *Nat Genet* 51 (2019), DOI 10.1038/s41588-019-0538-0
- Gasperini M et al., *Cell* 176 (2019), DOI 10.1016/j.cell.2018.11.029

…and the models you evaluate (Enformer / Borzoi / Cerberus / NTv3 / AlphaGenome /
AlphaGenome 2).

## License

MIT. See [LICENSE](LICENSE).
