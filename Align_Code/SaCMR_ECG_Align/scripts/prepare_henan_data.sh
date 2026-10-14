#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
cd "$PROJECT_DIR"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$PYTHON" -u -m SaCMR_ECG_Align.data.prepare_henan_splits \
  --sa_root "${SA_ROOT:-data/Henan_SA_processed_x112_y10_cinema}" "$@"
