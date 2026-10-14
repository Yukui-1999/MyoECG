#!/usr/bin/env bash
set -euo pipefail

SACMR_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SACMR_PYTHON="${SACMR_PYTHON:-python}"
cd "$SACMR_PROJECT_ROOT"

printf '[SaCMR] Starting Python; startup and data indexing progress will follow.\n'
exec "$SACMR_PYTHON" -u -m SaCMR_encoder.main_pretrain \
  --data_root "$SACMR_PROJECT_ROOT/data/processed_sa_x112_y10" \
  --train_manifest "$SACMR_PROJECT_ROOT/SaCMR_encoder/manifests/train_manifest_nvme.txt" \
  --output_dir "$SACMR_PROJECT_ROOT/SaCMR_encoder/outputs/mae_vit_base_patch16_validated" \
  --device cuda:3 \
  --model mae_vit_base_patch16 \
  --batch_size 256 \
  --accum_iter 1 \
  --epochs 400 \
  --warmup_epochs 40 \
  --mask_ratio 0.75 \
  --num_workers 16 \
  --prefetch_factor 1 \
  --decoder_slice_chunk_size 2 \
  "$@"
