#!/usr/bin/env bash
#SBATCH --job-name=cmnist-profile
#SBATCH --account=gcnu-ac
#SBATCH --partition=gpu_lowp
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/cmnist-profile-%j.out
#SBATCH --error=logs/cmnist-profile-%j.err

set -eo pipefail

REPO_ROOT=/nfs/ghome/live/danceh/DeconfoundingFlowMatching
CONDA_ROOT=/nfs/ghome/live/danceh/.local/mambaforge
RESULT_DIR="${REPO_ROOT}/logs/cmnist-profile-${SLURM_JOB_ID}"
GPU_LOG="${REPO_ROOT}/logs/cmnist-profile-${SLURM_JOB_ID}-gpu.csv"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate deconfoundingfm
set -u
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME}"
echo "result_dir=${RESULT_DIR}"
python --version
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=5 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

/usr/bin/time -v python examples/cmnist/run.py \
  --device cuda \
  --output "${RESULT_DIR}" \
  --propensity-trees 50 \
  --propensity-cv-folds 2 \
  --nuisance-steps 100 \
  --target-steps 100 \
  --checkpoints 25 50 100 \
  --eval-n 16 \
  --checkpoint-eval-n 8 \
  --sw2-projections 16 \
  --sample-chunk 8

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

echo "profile_complete=1"
