#!/bin/bash
# =============================================================================
# SLURM array job: AlphaGenome RNA-Seq with Karollus-matched test-time
# augmentation (3 shifts {-43,0,+43} bp × forward+reverse-complement = 6 passes),
# sharded over enhancer-gene pairs.
#
# Each array task processes pairs[shard_idx::num_shards] and writes one
# per-shard CSV. After the array finishes, run the merge step (bottom of file)
# to concatenate shards and compute the final correlations.
#
# ---- Before submitting -------------------------------------------------------
#  1. Edit SBATCH --partition/--qos/--gres to match your cluster.
#  2. Set NUM_SHARDS = array size. Array indices must be 0..NUM_SHARDS-1.
#  3. The genome FASTA is downloaded to ./.cache on first use. With many shards
#     racing to download it, pre-populate the cache ONCE first, e.g.:
#         python -c "from crispri_eval.dataset_utils import get_genome; get_genome('hg38')"
#     (or symlink an existing ./.cache from a previous run). Then submit.
#  4. The exon cache (metadata/{Fulco,Gasperini}_gene_exons_hg38.csv) ships
#     complete, so shards do NOT write to it concurrently. If you point
#     --exon_csv elsewhere and it is incomplete, pre-build it once first.
# =============================================================================

#SBATCH --job-name=ag_tta
#SBATCH --output=out/ag_tta_%A_%a.out
#SBATCH --error=out/ag_tta_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --partition=
#SBATCH --qos=
#SBATCH --array=0-7        # NUM_SHARDS - 1  (8 shards here)

set -euo pipefail

NUM_SHARDS=8
DATASET=${DATASET:-fulco}          # "fulco" or "gasperini" (override: DATASET=gasperini sbatch ...)
REPO=/grid/koo/home/amurphy/projects/seq2func_crispri_eval
VENV=/grid/koo/home/amurphy/projects/alphagenome-pytorch/.venv

source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONPATH=.

# Expandable allocator helps with the 1 Mb / 1 bp activation footprint.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:+$PYTORCH_CUDA_ALLOC_CONF,}expandable_segments:True"

SAVE_PATH="./results/test_${DATASET^}_alphagenome_tta/"

python scripts/test_${DATASET}_alphagenome.py \
    --modality rna_seq \
    --tta_shifts=-43,0,43 \
    --tta_rev_comp \
    --num_shards "${NUM_SHARDS}" \
    --shard_idx "${SLURM_ARRAY_TASK_ID}" \
    --save_path "${SAVE_PATH}"

# =============================================================================
# After all array tasks finish, merge (run once, CPU is fine):
#
#   # Fulco:
#   python scripts/merge_tta_shards.py \
#       --shards './results/test_Fulco_alphagenome_tta/Fulco_AlphaGenome_base_rna_seq_shard*of008_results.csv' \
#       --output_prefix Fulco_AlphaGenome_base_rna_seq_tta \
#       --fulco_corr_observed_subset knockdown_only
#
#   # Gasperini (no observed-subset flag — uses all pairs):
#   python scripts/merge_tta_shards.py \
#       --shards './results/test_Gasperini_alphagenome_tta/Gasperini_AlphaGenome_base_rna_seq_shard*of008_results.csv' \
#       --output_prefix Gasperini_AlphaGenome_base_rna_seq_tta
# =============================================================================
