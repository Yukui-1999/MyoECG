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

cd "${PROJECT_DIR}" || exit 1
export CUDA_VISIBLE_DEVICES=${GPU}

"${PYTHON}" -u main_align_sem_stage1UKB.py \
    --cmr_teacher_path "${CMR_TEACHER_PATH}" \
    --ecg_config_path "${PROJECT_DIR}/SaCMR_ECG_Align/configs/ecg_sem_stage1.yaml" \
    --device ${DEVICE} \
    --batch_size 256 \
    --test_batch_size 128 \
    --blr 0.001 \
    --epochs 100 \
    --warmup_epochs 20 \
    --seed 42 \
    --sem_loss_weight 1.0 \
    --pheno_loss_weight 1.0 \
    --rel_ecg_weight 0.5 \
    --exp_name "ecg_sem_stage1UKB_${NOW}" \
    --output_dir "outputs/ECG_CMR" \
    ${PY_ARGS}
