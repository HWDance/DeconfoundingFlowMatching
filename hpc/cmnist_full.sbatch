#!/usr/bin/env bash
#SBATCH --job-name=cmnist-full
#SBATCH --account=gcnu-ac
#SBATCH --partition=gpu_lowp
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cmnist-full-%j.out
#SBATCH --error=logs/cmnist-full-%j.err

set -eo pipefail

REPO_ROOT=/nfs/ghome/live/danceh/DeconfoundingFlowMatching
CONDA_ROOT=/nfs/ghome/live/danceh/.local/mambaforge
RESULT_DIR="${REPO_ROOT}/examples/cmnist/results/default"
GPU_LOG="${REPO_ROOT}/logs/cmnist-full-${SLURM_JOB_ID}-gpu.csv"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate deconfoundingfm
set -u
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME}"
echo "result_dir=${RESULT_DIR}"
python --version
python -c "import torch, torchvision; print(f'torch={torch.__version__} torchvision={torchvision.__version__}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=30 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

/usr/bin/time -v python examples/cmnist/run.py \
  --device cuda \
  --output "${RESULT_DIR}"

python examples/cmnist/audit_results.py \
  "${RESULT_DIR}" \
  --device cuda \
  --require-full-defaults

python -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  examples/cmnist/demo.ipynb

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

echo "full_run_and_notebook_complete=1"
