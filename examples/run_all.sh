#!/usr/bin/env bash
# =============================================================================
# Run all 8 (dataset x model) evaluations + 2 combined plots, end to end.
#
# Assumes:
#   - The crispri_eval package is installed (pip install -e .)
#   - Model-specific dependencies are installed (see requirements.txt)
#   - A CUDA-capable GPU is available; falls back to CPU if not
#
# Expect ~3-6 hours on a single A100 for the full run. To shortcut, comment
# out the entries you do not need.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_ROOT=./results
PLOTS_DIR=${RESULTS_ROOT}/plots
mkdir -p "${PLOTS_DIR}"

# -----------------------------------------------------------------------------
# Fulco
# -----------------------------------------------------------------------------

python scripts/test_fulco_enformer.py     --save_path ${RESULTS_ROOT}/test_Fulco_enformer/
python scripts/test_fulco_borzoi.py       --save_path ${RESULTS_ROOT}/test_Fulco_borzoi/    --use_flashzoi
python scripts/test_fulco_ntv3.py         --save_path ${RESULTS_ROOT}/test_Fulco_ntv3/      --autocast_dtype bf16
python scripts/test_fulco_alphagenome.py  --save_path ${RESULTS_ROOT}/test_Fulco_alphagenome/

# -----------------------------------------------------------------------------
# Gasperini
# -----------------------------------------------------------------------------

python scripts/test_gasperini_enformer.py     --save_path ${RESULTS_ROOT}/test_Gasperini_enformer/
python scripts/test_gasperini_borzoi.py       --save_path ${RESULTS_ROOT}/test_Gasperini_borzoi/    --use_flashzoi
python scripts/test_gasperini_ntv3.py         --save_path ${RESULTS_ROOT}/test_Gasperini_ntv3/      --autocast_dtype bf16
python scripts/test_gasperini_alphagenome.py  --save_path ${RESULTS_ROOT}/test_Gasperini_alphagenome/

# -----------------------------------------------------------------------------
# Plots (filenames assume the default --output_prefix from each script)
# -----------------------------------------------------------------------------

python scripts/plot_fulco_results.py \
    --results \
        ${RESULTS_ROOT}/test_Fulco_enformer/Fulco_base_enformer_results.csv \
        ${RESULTS_ROOT}/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv \
        ${RESULTS_ROOT}/test_Fulco_ntv3/Fulco_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
        ${RESULTS_ROOT}/test_Fulco_alphagenome/Fulco_AlphaGenome_base_rna_seq_results.csv \
    --labels "Enformer" "Borzoi" "NTv3" "AlphaGenome" \
    --max_width_subplots 4 \
    --observed_mode decrease_only \
    --output ${PLOTS_DIR}/fulco_all_models.png

python scripts/plot_gasperini_results.py \
    --results \
        ${RESULTS_ROOT}/test_Gasperini_enformer/Gasperini_base_enformer_results.csv \
        ${RESULTS_ROOT}/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv \
        ${RESULTS_ROOT}/test_Gasperini_ntv3/Gasperini_NTv3_InstaDeepAI_NTv3_650M_post_rna_results.csv \
        ${RESULTS_ROOT}/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv \
    --labels "Enformer" "Borzoi" "NTv3" "AlphaGenome" \
    --max_width_subplots 4 \
    --output ${PLOTS_DIR}/gasperini_all_models.png

echo "Done. Plots: ${PLOTS_DIR}/"
