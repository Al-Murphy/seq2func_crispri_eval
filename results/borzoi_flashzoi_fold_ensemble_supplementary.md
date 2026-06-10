# Supplementary: Borzoi/Flashzoi fold ensemble on CRISPRi enhancer–gene benchmarks

Borzoi/Flashzoi ships four independently trained replicates (folds 0–3). Below,
each fold is scored individually on the Fulco (2019) and Gasperini (2019) CRISPRi
enhancer-knockdown benchmarks (base Flashzoi backbone, K562 RNA-Seq tracks), and
compared against the **4-fold ensemble** — the mean of the four replicates'
predictions (`pred_wt`, `pred_crispri_mean` averaged, then
`pred_delta = (pred_wt − pred_crispri_mean) / (pred_wt + 1e-8)` recomputed). For
context, Enformer, NTv3, and AlphaGenome (single base models) are also shown.

Correlations are between the observed CRISPRi effect (`y_delta`) and predicted
effect (`pred_delta`). `log2_*` columns are the secondary Karollus-style metric:
−log₂(1 − y_delta) vs log₂(pred_wt / pred_crispri).

## Fulco (knockdown-only rows, `y_delta > 0`)

| Model | n | Pearson r | Spearman ρ | log2 Pearson r | log2 Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| Enformer | 62 | 0.2923 | 0.1891 | — | — |
| NTv3 | 63 | 0.3381 | 0.0908 | 0.3706 | 0.0908 |
| AlphaGenome | 63 | 0.6739 | 0.5403 | 0.6489 | 0.5403 |
| Borzoi fold 0 | 63 | 0.6564 | 0.5219 | 0.6643 | 0.5219 |
| Borzoi fold 1 | 63 | 0.5514 | 0.3237 | 0.5231 | 0.3237 |
| Borzoi fold 2 | 63 | 0.6453 | 0.3222 | 0.6776 | 0.3222 |
| Borzoi fold 3 | 63 | 0.5784 | 0.4862 | 0.6402 | 0.4862 |
| **Borzoi 4-fold ensemble** | 63 | **0.6608** | **0.4863** | **0.6803** | **0.4863** |

## Gasperini (all high-confidence rows)

| Model | n | Pearson r | Spearman ρ | log2 Pearson r | log2 Spearman ρ |
| --- | --- | --- | --- | --- | --- |
| Enformer | 352 | 0.0346 | 0.0383 | — | — |
| NTv3 | 438 | 0.1206 | 0.1367 | 0.0741 | 0.1367 |
| AlphaGenome | 438 | 0.4461 | 0.4524 | 0.2905 | 0.4524 |
| Borzoi fold 0 | 417 | 0.2958 | 0.2723 | 0.2127 | 0.2723 |
| Borzoi fold 1 | 417 | 0.3328 | 0.3073 | 0.2385 | 0.3073 |
| Borzoi fold 2 | 417 | 0.2721 | 0.2926 | 0.2165 | 0.2926 |
| Borzoi fold 3 | 417 | 0.2859 | 0.2706 | 0.2033 | 0.2706 |
| **Borzoi 4-fold ensemble** | 417 | **0.3392** | **0.3253** | **0.2496** | **0.3253** |

**Takeaway.** On both benchmarks the 4-fold ensemble matches or beats every
individual Borzoi replicate (highest or tied-highest Pearson r on each), and on
the log2 metric exceeds all single folds — the expected benefit of averaging the
replicate models. Single-fold numbers also show the run-to-run spread across folds.

## Reproduce

```bash
# 1. Per-fold runs: folds 0 (default) + 1-3 via the job-array scripts
#    job_scripts/clg_fulco_flashzoi_rna_folds.sh, clg_gasperini_flashzoi_rna_folds.sh

# 2. Build the 4-fold ensemble result CSVs
python scripts/3_benchmark/make_borzoi_ensemble.py \
  --inputs ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \
           ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_rep{1,2,3}_results.csv \
  --output ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv

# 3. Comparison plots (Borzoi entry = the ensemble CSV)
python scripts/3_benchmark/plot_fulco_results.py \
  --results ./results/test_Fulco_enformer/Fulco_base_base_enformer_results.csv \
            ./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv \
            ./results/test_Fulco_ntv3/Fulco_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
            ./results/test_Fulco_alphagenome/Fulco_AlphaGenome_base_rna_seq_results.csv \
  --labels "Enformer" "Borzoi (4-fold ensemble)" "NTv3" "AlphaGenome" \
  --max_width_subplots 4 --observed_mode decrease_only \
  --output ./results/test_Fulco_alphagenome/plots/fulco_enf_borzoiens_ntv3_ag.png

# 4. This summary table
python scripts/3_benchmark/summarize_benchmark_correlations.py --benchmark fulco \
  --results "Borzoi 4-fold ensemble=./results/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_ensemble_results.csv" ...
```
