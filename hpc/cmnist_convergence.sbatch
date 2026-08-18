#!/usr/bin/env bash
#SBATCH --job-name=cmnist-convergence
#SBATCH --account=gcnu-ac
#SBATCH --partition=gpu_lowp
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/cmnist-convergence-%j.out
#SBATCH --error=logs/cmnist-convergence-%j.err

set -eo pipefail

REPO_ROOT=/nfs/ghome/live/danceh/DeconfoundingFlowMatching
CONDA_ROOT=/nfs/ghome/live/danceh/.local/mambaforge
RESULT_DIR="${REPO_ROOT}/examples/cmnist/results/default"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate deconfoundingfm
set -u
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

python examples/cmnist/recompute_checkpoint_convergence.py \
  "${RESULT_DIR}" \
  --device cuda

python examples/cmnist/audit_results.py \
  "${RESULT_DIR}" \
  --device cuda \
  --require-full-defaults

python -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  examples/demo_cmnist_online.ipynb

echo "shared_checkpoint_convergence_complete=1"
