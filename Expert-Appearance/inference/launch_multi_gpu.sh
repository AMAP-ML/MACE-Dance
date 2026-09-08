#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --csv-path INPUT.csv --save-dir OUTPUT_DIR --gpus GPU_IDS [-- PYTHON_ARGS]"
    echo "GPU_IDS is a comma-separated list such as 0 or 0,1,2,3."
    exit "${1:-1}"
}

CSV_PATH=""
SAVE_DIR=""
GPUS=""
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv-path) CSV_PATH="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --) shift; PYTHON_ARGS=("$@"); break ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$CSV_PATH" || -z "$SAVE_DIR" || -z "$GPUS" ]]; then
    usage
fi
if [[ ! -f "$CSV_PATH" ]]; then
    echo "CSV file not found: $CSV_PATH" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/appearance_infer.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

NUM_DATA_LINES=$(tail -n +2 "$CSV_PATH" | wc -l)
if [[ "$NUM_DATA_LINES" -eq 0 ]]; then
    echo "No data rows in $CSV_PATH; nothing to do."
    exit 0
fi

IFS=',' read -r -a RAW_GPU_IDS <<< "$GPUS"
GPU_IDS=()
for GPU_ID in "${RAW_GPU_IDS[@]}"; do
    GPU_ID="${GPU_ID//[[:space:]]/}"
    [[ -n "$GPU_ID" ]] && GPU_IDS+=("$GPU_ID")
done
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "--gpus did not contain any GPU IDs" >&2
    exit 1
fi
if [[ "${#GPU_IDS[@]}" -gt "$NUM_DATA_LINES" ]]; then
    GPU_IDS=("${GPU_IDS[@]:0:$NUM_DATA_LINES}")
fi

NUM_GPUS=${#GPU_IDS[@]}
mkdir -p "$SAVE_DIR"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Rows: $NUM_DATA_LINES; GPUs: ${GPU_IDS[*]}; output: $SAVE_DIR"
PIDS=()
for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    START_INDEX=$(( i * NUM_DATA_LINES / NUM_GPUS ))
    END_INDEX=$(( (i + 1) * NUM_DATA_LINES / NUM_GPUS ))
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "$PYTHON_SCRIPT" \
        --csv_path "$CSV_PATH" \
        --save_dir "$SAVE_DIR" \
        --start-index "$START_INDEX" \
        --end-index "$END_INDEX" \
        "${PYTHON_ARGS[@]}" &
    PIDS+=("$!")
done

EXIT_CODE=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        EXIT_CODE=1
    fi
done
if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "One or more inference shards failed." >&2
    exit "$EXIT_CODE"
fi
echo "All inference shards completed: $SAVE_DIR"
