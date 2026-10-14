#!/usr/bin/env bash

set -x

PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-"python"}
GPU=${GPU:-3}
DEVICE=${DEVICE:-"cuda:0"}
NOW=${NOW:-$(date +"%Y%m%d_%H%M%S")}

if [[ -z "${CMR_TEACHER_PATH:-}" ]]; then
    echo "CMR_TEACHER_PATH must point to best-cmr-pheno.pth" >&2
    exit 1
fi
if [[ -z "${ECG_SEM_STAGE1_CONFIG_PATH:-}" ]]; then
    echo "ECG_SEM_STAGE1_CONFIG_PATH must point to st_mem_aligned_ecg_sem_stage1.yaml" >&2
    exit 1
fi

cd "${PROJECT_DIR}" || exit 1
export CUDA_VISIBLE_DEVICES=${GPU}

"${PYTHON}" -u main_align_sem_stage2FCCCH.py \
    --cmr_teacher_path "${CMR_TEACHER_PATH}" \
    --ecg_config_path "${ECG_SEM_STAGE1_CONFIG_PATH}" \
    --device ${DEVICE} \
    --batch_size 64 \
    --test_batch_size 100 \
    --blr 0.0001 \
    --epochs 50 \
    --warmup_epochs 5 \
    --seed 42 \
    --sem_loss_weight 1.0 \
    --exp_name "ecg_sem_stage2FCCCH_${NOW}" \
    --output_dir "outputs/ECG_CMR" \
    ${PY_ARGS}
