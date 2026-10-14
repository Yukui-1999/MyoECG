#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
GPU=${GPU:-3}
DEVICE=${DEVICE:-cuda:0}
EXP_NAME=${EXP_NAME:-ecg_sem_stage1}
CMR_TEACHER_PATH=${CMR_TEACHER_PATH:-$PROJECT_DIR/SaCMR_ECG_Align/outputs/cmr_sem_teacher/best-cmr-pheno.pth}
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$PYTHON" -u -m SaCMR_ECG_Align.main_align_sem_stage1 \
  --device "$DEVICE" --cmr_teacher_path "$CMR_TEACHER_PATH" \
  --batch_size "${BATCH_SIZE:-128}" --test_batch_size 64 \
  --num_workers "${NUM_WORKERS:-4}" --prefetch_factor "${PREFETCH_FACTOR:-1}" \
  --blr 0.001 --epochs 100 --warmup_epochs 20 --seed 42 \
  --sem_loss_weight 1.0 --pheno_loss_weight 1.0 --rel_ecg_weight 0.5 \
  --exp_name "$EXP_NAME" "$@"
