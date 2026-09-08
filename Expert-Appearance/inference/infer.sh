#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV_PATH="${1:-${APP_ROOT}/test_data.csv}"
SAVE_DIR="${2:-${APP_ROOT}/outputs}"
GPUS="${3:-0}"
if [[ "$#" -gt 0 ]]; then shift; fi
if [[ "$#" -gt 0 ]]; then shift; fi
if [[ "$#" -gt 0 ]]; then shift; fi

export PYTHONPATH="${APP_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash "${SCRIPT_DIR}/launch_multi_gpu.sh" \
  --csv-path "$CSV_PATH" \
  --save-dir "$SAVE_DIR" \
  --gpus "$GPUS" \
  -- \
  --base_model_dir "${BASE_MODEL_DIR:-${APP_ROOT}/models/Wan-AI/Wan2.2-Animate-14B}" \
  --adapter_path "${ADAPTER_PATH:-${APP_ROOT}/models/train/Adapter.safetensors}" \
  --lora_path "${LORA_PATH:-${APP_ROOT}/models/train/LoRA.safetensors}" \
  --height "${HEIGHT:-1280}" --width "${WIDTH:-720}" \
  --num_frames "${NUM_FRAMES:-81}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-20}" \
  --cfg_scale "${CFG_SCALE:-1.0}" \
  --fps "${FPS:-15}" --quality "${QUALITY:-5}" \
  "$@"
