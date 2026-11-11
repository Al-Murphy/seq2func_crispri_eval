# Supplementary: Borzoi/Flashzoi fold ensemble on CRISPRi enhancer–gene benchmarks

Borzoi/Flashzoi ships four independently trained replicates (folds 0–3). Below,
each fold is scored individually on the Fulco (2019) and Gasperini (2019) CRISPRi
enhancer-knockdown benchmarks (base Flashzoi backbone, bf16, K562 RNA-Seq tracks),
and compared against the **4-fold ensemble**: the mean of the four replicates'
predictions (`pred_wt`, `pred_crispri_mean` averaged, then
`pred_delta = (pred_wt − pred_crispri_mean) / (pred_wt + 1e-8)` recomputed). The
other models are shown for context (Cerberus is itself an 8-replicate ensemble).

Borzoi and Cerberus use the exon readout (README, "Gene readouts"): strand-matched
K562 RNA-Seq tracks, converted back to coverage with Calico's untransform and
summed over the output bins overlapping the gene's exons. The "(TSS window)" rows
are the earlier readout (all tracks, transformed units, 21 × 32 bp bins centred on
the TSS), kept for comparison.

Correlations are between the observed CRISPRi effect (`y_delta`) and predicted
effect (`pred_delta`). `log2_*` columns are the secondary Karollus-style metric:
−log₂(1 − y_delta) vs log₂(pred_wt / pred_crispri). Each model is scored on every
pair inside its own input window, so `n` differs between models.

## Fulco (knockdown-only rows, `y_delta > 0`)

| Model | n | Pearson r | Spearman ρ | log2 Pearson r | log2 Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| Enformer | 62 | 0.5981 | 0.3231 | — | — |
| Borzoi fold 0 | 63 | 0.5103 | 0.5378 | 0.4980 | 0.5378 |
| Borzoi fold 1 | 63 | 0.4894 | 0.3282 | 0.4483 | 0.3282 |
| Borzoi fold 2 | 63 | 0.5582 | 0.2369 | 0.5946 | 0.2369 |
| Borzoi fold 3 | 63 | 0.5555 | 0.4427 | 0.6027 | 0.4427 |
| **Borzoi 4-fold ensemble** | 63 | **0.6389** | **0.4565** | **0.6630** | **0.4565** |
| Borzoi 4-fold ensemble (TSS window) | 63 | 0.6608 | 0.4863 | 0.6803 | 0.4863 |
| Cerberus | 63 | 0.7033 | 0.5893 | 0.7129 | 0.5893 |
| Cerberus (TSS window) | 63 | 0.6390 | 0.3898 | 0.6579 | 0.3898 |
| NTv3 | 63 | 0.5113 | 0.2657 | 0.5233 | 0.2657 |
| AlphaGenome | 63 | 0.6739 | 0.5403 | 0.6489 | 0.5403 |

## Gasperini (all high-confidence rows)

| Model | n | Pearson r | Spearman ρ | log2 Pearson r | log2 Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| Enformer | 352 | 0.2964 | 0.2723 | — | — |
| Borzoi fold 0 | 417 | 0.3148 | 0.2881 | 0.1997 | 0.2881 |
| Borzoi fold 1 | 417 | 0.3516 | 0.3172 | 0.2377 | 0.3172 |
| Borzoi fold 2 | 417 | 0.2611 | 0.2974 | 0.2028 | 0.2974 |
| Borzoi fold 3 | 417 | 0.3207 | 0.3295 | 0.2164 | 0.3295 |
| **Borzoi 4-fold ensemble** | 417 | **0.3679** | **0.3561** | **0.2478** | **0.3561** |
| Borzoi 4-fold ensemble (TSS window) | 417 | 0.3392 | 0.3253 | 0.2496 | 0.3253 |
| Cerberus | 427 | 0.3405 | 0.3060 | 0.2717 | 0.3060 |
| Cerberus (TSS window) | 427 | 0.2611 | 0.2661 | 0.2385 | 0.2661 |
| NTv3 | 438 | 0.2916 | 0.2930 | 0.2299 | 0.2930 |
| AlphaGenome | 438 | 0.4461 | 0.4524 | 0.2905 | 0.4524 |

**Takeaway.** The 4-fold ensemble beats every individual Borzoi replicate on
Pearson r on both benchmarks, the expected benefit of averaging replicate models;
on Spearman ρ it is best on Gasperini but below fold 0 on Fulco. Single-fold
numbers show the spread across replicates, which on Fulco's 63 pairs is large
for Spearman ρ (0.24–0.54). That spread is the replicates themselves, not the
dinucleotide shuffles. These runs predate seeding, so each drew its own shuffles,
but redrawing them moves a single replicate's Fulco ρ by only SD 0.010–0.022, and
the ensemble's by 0.005 (README, "Shuffles").

## Reproduce

```bash
# 1. Per-fold runs (replicate 0 = script default; 1-3 via --backbone_replicate)
for k in 0 1 2 3; do
  python scripts/test_gasperini_borzoi.py --use_flashzoi \
    --backbone_replicate johahi/borzoi-replicate-$k \
    --output_prefix Gasperini_Borzoi_base_flashzoi_rna$([ $k -gt 0 ] && echo _rep$k) --debug_export
done   # likewise scripts/test_fulco_borzoi.py

# 2. Build the 4-fold ensemble result CSVs
python scripts/make_borzoi_ensemble.py \
  --inputs ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \
           ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_rep{1,2,3}_results.csv \
  --output ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv

# 3. This summary table
python scripts/summarize_benchmark_correlations.py --benchmark fulco \
  --results "Borzoi 4-fold ensemble=./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv" ... \
  --output_csv ./results/test_Fulco_borzoi/fulco_correlation_summary.csv
```
