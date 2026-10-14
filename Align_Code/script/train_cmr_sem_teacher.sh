#!/usr/bin/env bash

set -x

PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-"python"}
GPU=${GPU:-3}
DEVICE=${DEVICE:-"cuda:0"}
NOW=${NOW:-$(date +"%Y%m%d_%H%M%S")}

cd "${PROJECT_DIR}" || exit 1
export CUDA_VISIBLE_DEVICES=${GPU}

"${PYTHON}" -u main_train_cmr_sem_teacher.py \
    --device ${DEVICE} \
    --batch_size 128 \
    --test_batch_size 128 \
    --blr 0.001 \
    --epochs 100 \
    --warmup_epochs 20 \
    --freeze_encoder_epochs 50 \
    --unfreeze_last_blocks 2 \
    --seed 42 \
    --exp_name "cmr_sem_teacher_${NOW}" \
    --output_dir "outputs/ECG_CMR" \
    ${PY_ARGS}
