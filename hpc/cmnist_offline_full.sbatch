#!/usr/bin/env bash
#SBATCH --job-name=cmnist-offline-100k
#SBATCH --account=gcnu-ac
#SBATCH --partition=gpu_lowp
#SBATCH --qos=normal
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cmnist-offline-100k-%j.out
#SBATCH --error=logs/cmnist-offline-100k-%j.err

set -eo pipefail

REPO_ROOT=/nfs/ghome/live/danceh/DeconfoundingFlowMatching
CONDA_ROOT=/nfs/ghome/live/danceh/.local/mambaforge
FINAL_RESULT_DIR="${REPO_ROOT}/examples/cmnist/results/offline"
STAGING_ROOT="${REPO_ROOT}/logs/cmnist-offline-refresh-${SLURM_JOB_ID}"
RESULT_DIR="${STAGING_ROOT}/new-offline"
BACKUP_DIR="${STAGING_ROOT}/previous-offline"
GPU_LOG="${REPO_ROOT}/logs/cmnist-offline-100k-${SLURM_JOB_ID}-gpu.csv"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate deconfoundingfm
set -u
cd "${REPO_ROOT}"
mkdir -p "${STAGING_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME}"
echo "staging_result_dir=${RESULT_DIR}"
echo "final_result_dir=${FINAL_RESULT_DIR}"
python --version
python -c "import torch, torchvision; print(f'torch={torch.__version__} torchvision={torchvision.__version__}')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nvidia-smi \
  --query-gpu=timestamp,index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=30 > "${GPU_LOG}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

/usr/bin/time -v python examples/cmnist/run_offline.py \
  --device cuda \
  --output "${RESULT_DIR}"

python examples/cmnist/audit_results.py \
  "${RESULT_DIR}" \
  --device cuda \
  --require-full-defaults \
  --expected-study-mode offline_empirical

if [[ -e "${BACKUP_DIR}" ]]; then
  echo "Refusing to overwrite existing staging backup: ${BACKUP_DIR}" >&2
  exit 1
fi
if [[ -e "${FINAL_RESULT_DIR}" ]]; then
  mv "${FINAL_RESULT_DIR}" "${BACKUP_DIR}"
fi
if ! mv "${RESULT_DIR}" "${FINAL_RESULT_DIR}"; then
  if [[ -e "${BACKUP_DIR}" && ! -e "${FINAL_RESULT_DIR}" ]]; then
    mv "${BACKUP_DIR}" "${FINAL_RESULT_DIR}"
  fi
  exit 1
fi

env CMNIST_OFFLINE_RESULT_DIR="${FINAL_RESULT_DIR}" python -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1200 \
  examples/cmnist/demo_offline.ipynb

if [[ -e "${BACKUP_DIR}" ]]; then
  rm -rf -- "${BACKUP_DIR}"
fi
rmdir "${STAGING_ROOT}" 2>/dev/null || true

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

echo "full_run_audit_replace_and_notebook_complete=1"
