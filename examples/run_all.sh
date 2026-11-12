#!/usr/bin/env bash
# =============================================================================
# Run all 12 (dataset x model) evaluations + 2 combined plots, end to end.
#
# Assumes:
#   - The crispri_eval package is installed (pip install -e .)
#   - Model-specific dependencies are installed (see requirements.txt)
#   - A CUDA-capable GPU is available; falls back to CPU if not
#   - Cerberus runs in its own baskerville environment (see README) with
#     CERBERUS_DIR=/path/to/baskerville/releases/cerberus. In fp32 it takes
#     ~3 min per pair on an H100 (~3 h Fulco, ~21 h Gasperini), so on a
#     cluster run it sharded instead (--num_shards / --shard_idx).
#   - AlphaGenome 2 runs through the hosted AlphaGenome API (CPU only) and is
#     skipped unless an API key is saved in ${AG2_KEY} (see README). Rate limits
#     make it ~1.4 h for Fulco; shard Gasperini (--num_shards) on a cluster.
#
# Expect ~3-6 hours on a single A100 for the other four models. To shortcut, comment
# out the entries you do not need.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_ROOT=./results
CERBERUS_DIR=${CERBERUS_DIR:-./models/cerberus}   # baskerville releases/cerberus
PLOTS_DIR=${RESULTS_ROOT}/plots
AG2_KEY=${AG2_KEY:-$HOME/.config/alphagenome/api_key}
mkdir -p "${PLOTS_DIR}"

# -----------------------------------------------------------------------------
# Fulco
# -----------------------------------------------------------------------------

python scripts/test_fulco_enformer.py     --save_path ${RESULTS_ROOT}/test_Fulco_enformer/
python scripts/test_fulco_borzoi.py       --save_path ${RESULTS_ROOT}/test_Fulco_borzoi/    --use_flashzoi
python scripts/test_fulco_cerberus.py     --save_path ${RESULTS_ROOT}/test_Fulco_cerberus/  --cerberus_dir ${CERBERUS_DIR}
python scripts/test_fulco_ntv3.py         --save_path ${RESULTS_ROOT}/test_Fulco_ntv3/
python scripts/test_fulco_alphagenome.py  --save_path ${RESULTS_ROOT}/test_Fulco_alphagenome/
if [[ -f "${AG2_KEY}" ]]; then
  python scripts/test_fulco_alphagenome.py --backend api --api_key_file "${AG2_KEY}" --per_track_export \
      --save_path ${RESULTS_ROOT}/test_Fulco_alphagenome/
fi

# -----------------------------------------------------------------------------
# Gasperini
# -----------------------------------------------------------------------------

python scripts/test_gasperini_enformer.py     --save_path ${RESULTS_ROOT}/test_Gasperini_enformer/
python scripts/test_gasperini_borzoi.py       --save_path ${RESULTS_ROOT}/test_Gasperini_borzoi/    --use_flashzoi
python scripts/test_gasperini_cerberus.py     --save_path ${RESULTS_ROOT}/test_Gasperini_cerberus/  --cerberus_dir ${CERBERUS_DIR}
python scripts/test_gasperini_ntv3.py         --save_path ${RESULTS_ROOT}/test_Gasperini_ntv3/
python scripts/test_gasperini_alphagenome.py  --save_path ${RESULTS_ROOT}/test_Gasperini_alphagenome/
if [[ -f "${AG2_KEY}" ]]; then
  python scripts/test_gasperini_alphagenome.py --backend api --api_key_file "${AG2_KEY}" --per_track_export \
      --save_path ${RESULTS_ROOT}/test_Gasperini_alphagenome/
fi

# -----------------------------------------------------------------------------
# Plots (filenames assume the default --output_prefix from each script)
# -----------------------------------------------------------------------------

# AlphaGenome 2 is added when its results exist.
LABELS=("Enformer" "Borzoi" "Cerberus" "NTv3" "AlphaGenome")
FULCO=(
    ${RESULTS_ROOT}/test_Fulco_enformer/Fulco_base_enformer_results.csv
    ${RESULTS_ROOT}/test_Fulco_borzoi/Fulco_Borzoi_base_flashzoi_rna_results.csv
    ${RESULTS_ROOT}/test_Fulco_cerberus/Fulco_Cerberus_rna_results.csv
    ${RESULTS_ROOT}/test_Fulco_ntv3/Fulco_NTv3_InstaDeepAI_NTv3_tracks_1mb_rna_results.csv
    ${RESULTS_ROOT}/test_Fulco_alphagenome/Fulco_AlphaGenome_base_rna_seq_results.csv
)
GASPERINI=(
    ${RESULTS_ROOT}/test_Gasperini_enformer/Gasperini_base_enformer_results.csv
    ${RESULTS_ROOT}/test_Gasperini_borzoi/Gasperini_Borzoi_base_flashzoi_rna_results.csv
    ${RESULTS_ROOT}/test_Gasperini_cerberus/Gasperini_Cerberus_rna_results.csv
    ${RESULTS_ROOT}/test_Gasperini_ntv3/Gasperini_NTv3_InstaDeepAI_NTv3_tracks_1mb_rna_results.csv
    ${RESULTS_ROOT}/test_Gasperini_alphagenome/Gasperini_AlphaGenome_base_rna_seq_results.csv
)
AG2_FULCO=${RESULTS_ROOT}/test_Fulco_alphagenome/Fulco_AlphaGenome2_api_rna_seq_results.csv
AG2_GASPERINI=${RESULTS_ROOT}/test_Gasperini_alphagenome/Gasperini_AlphaGenome2_api_rna_seq_results.csv
if [[ -f "${AG2_FULCO}" && -f "${AG2_GASPERINI}" ]]; then
    LABELS+=("AlphaGenome 2"); FULCO+=("${AG2_FULCO}"); GASPERINI+=("${AG2_GASPERINI}")
fi

python scripts/plot_fulco_results.py \
    --results "${FULCO[@]}" \
    --labels "${LABELS[@]}" \
    --max_width_subplots ${#LABELS[@]} \
    --observed_mode decrease_only \
    --output ${PLOTS_DIR}/fulco_all_models.png

python scripts/plot_gasperini_results.py \
    --results "${GASPERINI[@]}" \
    --labels "${LABELS[@]}" \
    --max_width_subplots ${#LABELS[@]} \
    --output ${PLOTS_DIR}/gasperini_all_models.png

echo "Done. Plots: ${PLOTS_DIR}/"
