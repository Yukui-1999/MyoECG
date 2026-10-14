#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
GPU=${GPU:-3}
DEVICE=${DEVICE:-cuda:0}
EXP_NAME=${EXP_NAME:-ecg_sem_stage2_henan}
ECG_STAGE1_PATH=${ECG_STAGE1_PATH:-$PROJECT_DIR/SaCMR_ECG_Align/outputs/ecg_sem_stage1/best-loss.pth}
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
OPTIONAL_ARGS=()
if [[ -n "${CMR_TEACHER_PATH:-}" ]]; then
  OPTIONAL_ARGS+=(--cmr_teacher_path "$CMR_TEACHER_PATH")
fi
if [[ -n "${ECG_SEM_STAGE1_CONFIG_PATH:-}" ]]; then
  OPTIONAL_ARGS+=(--ecg_config_path "$ECG_SEM_STAGE1_CONFIG_PATH")
fi
exec "$PYTHON" -u -m SaCMR_ECG_Align.main_align_sem_stage2 \
  --device "$DEVICE" --ecg_stage1_path "$ECG_STAGE1_PATH" \
  --batch_size "${BATCH_SIZE:-64}" --test_batch_size "${TEST_BATCH_SIZE:-100}" \
  --num_workers "${NUM_WORKERS:-4}" --prefetch_factor "${PREFETCH_FACTOR:-1}" \
  --blr 0.0001 --epochs 50 --warmup_epochs 5 --seed 42 \
  --sem_loss_weight 1.0 --best_patience "${BEST_PATIENCE:-5}" \
  --exp_name "$EXP_NAME" "${OPTIONAL_ARGS[@]}" "$@"
