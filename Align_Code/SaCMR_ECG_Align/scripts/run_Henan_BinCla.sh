#!/usr/bin/env bash

# 河南心肌病二分类：从短轴 CMR 对齐后的 ECG 权重微调，单次运行。
# 参考 scripts/ECG/baseline/run_Henan_BinCla_NCVR_R1_5fold.sh。
# 任意一次训练失败就停止；pipefail 保证 tee 不会掩盖训练错误。
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
GPU=${GPU:-0}

# 保持原河南心肌病二分类的训练设置，可通过环境变量覆盖。
Batch_size=${BATCH_SIZE:-32}
Seed=${SEED:-42}
Lr=${LR:-5e-5}
Num_workers=${NUM_WORKERS:-16}
Input_modality='ECG'
Dis='cm'
Threshold_method='youden'

# 短轴 stage2 已完成的 ECG 模型；YAML 的 afterAlign_path 指向 best-loss.pth。
STAGE2_DIR=${STAGE2_DIR:-$PROJECT_DIR/SaCMR_ECG_Align/outputs/ecg_sem_stage2_henan}
Ecg_config_path="$STAGE2_DIR/st_mem_aligned_ecg_sem_stage2.yaml"
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/ECG_CMR_FCCCH}
NOW=$(date +"%Y%m%d_%H%M%S")

cd "$PROJECT_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

if [ ! -f "$Ecg_config_path" ] || [ ! -f "$STAGE2_DIR/best-loss.pth" ]; then
    echo "缺少短轴 Stage2 的 ECG 配置或权重：$STAGE2_DIR" >&2
    exit 1
fi

# 沿用 main_Cladownstream_HeNan.py 的原 seed_3407/train.xlsx、valid.xlsx、test.xlsx。
# 不重新分折，不使用短轴配对清单；疾病标签与混杂样本过滤也沿用原代码。
ECG_type='st_mem_SaCMR_aligned_ecg_sem_stage2'
CKPT_DIR="$OUTPUT_ROOT/ECG_CMR_base_Cla_bs${Batch_size}_lr${Lr}_seed${Seed}_${ECG_type}_Dis${Dis}_Thres${Threshold_method}_NoFiltering_${NOW}"
mkdir -p "$CKPT_DIR"

echo "开始河南心肌病训练：短轴对齐后的 ECG，GPU ${GPU}"
echo "预训练配置：$Ecg_config_path"
echo "结果目录：$CKPT_DIR"

"$PYTHON" -u main_Cladownstream_HeNan.py \
    --input_modality "$Input_modality" \
    --batch_size "$Batch_size" \
    --seed "$Seed" \
    --blr "$Lr" \
    --num_workers "$Num_workers" \
    --ecg_config_path "$Ecg_config_path" \
    --output_dir "$CKPT_DIR" \
    --ecg_model 'stmem' \
    --num_classes 1 \
    --dis "$Dis" \
    --threshold_method "$Threshold_method" \
    --test_dir_name 'test' \
    "$@" \
    2>&1 | tee -a "$CKPT_DIR/train-${NOW}.log"
