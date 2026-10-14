#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
GPU=${GPU:-3}
DEVICE=${DEVICE:-cuda:0}
EXP_NAME=${EXP_NAME:-cmr_sem_teacher}
cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$PYTHON" -u -m SaCMR_ECG_Align.main_train_cmr_sem_teacher \
  --device "$DEVICE" --batch_size "${BATCH_SIZE:-64}" --test_batch_size 64 \
  --num_workers "${NUM_WORKERS:-4}" --prefetch_factor "${PREFETCH_FACTOR:-1}" \
  --blr 0.001 --epochs 100 --warmup_epochs 20 \
  --freeze_encoder_epochs 50 --unfreeze_last_blocks 2 --seed 42 \
  --exp_name "$EXP_NAME" "$@"
