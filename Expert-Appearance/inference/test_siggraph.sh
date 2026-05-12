#!/bin/bash
set -e

usage() {
    echo "Usage: $0 --csv-path <path_to_csv> --save-dir <path_to_dir> --gpus <gpu_ids> [-- other_python_args]"
    echo ""
    echo "Arguments:"
    echo "  --csv-path      Path to input CSV."
    echo "  --save-dir      Directory to save generated videos."
    echo "  --gpus          Comma-separated GPU IDs, e.g. \"0,1,2,3\"."
    echo "  --              Separator. All following args are passed to python script."
    exit 1
}

PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv-path) CSV_PATH="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --) shift; PYTHON_ARGS=("$@"); break ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$CSV_PATH" ] || [ -z "$SAVE_DIR" ] || [ -z "$GPUS" ]; then
    echo "Error: --csv-path, --save-dir, and --gpus are required."
    usage
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found at $CSV_PATH"
    exit 1
fi

# 获取当前 shell 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 这里按你的项目结构回到项目根目录
# 如果 test_cvpr.sh 在 Expert-Appearance/inference/ 下，则 PROJECT_ROOT 是上一层
# 如果在更深目录，就要多 ../ 一点
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Python 推理脚本路径
PYTHON_SCRIPT="${PROJECT_ROOT}/examples/wanvideo/model_training/validate_zjs/Wan2.2-Animate-14B-test.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

NUM_DATA_LINES=$(tail -n +2 "$CSV_PATH" | wc -l)
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NUM_GPUS=${#GPU_IDS[@]}

if [ "$NUM_DATA_LINES" -eq 0 ]; then
    echo "Warning: No data lines found in $CSV_PATH. Exiting."
    exit 0
fi

LINES_PER_GPU=$(( (NUM_DATA_LINES + NUM_GPUS - 1) / NUM_GPUS ))

echo "--------------------------------------------------"
echo "Starting Parallel Inference (Index-based)"
echo "--------------------------------------------------"
echo "Project root: $PROJECT_ROOT"
echo "Python script: $PYTHON_SCRIPT"
echo "Total data rows: $NUM_DATA_LINES"
echo "Number of GPUs: $NUM_GPUS (${GPU_IDS[*]})"
echo "Approx. lines per GPU: $LINES_PER_GPU"
echo "Results will be saved in: $SAVE_DIR"
echo "Python script arguments: ${PYTHON_ARGS[*]}"
echo "--------------------------------------------------"

mkdir -p "$SAVE_DIR"

PIDS=()
for i in "${!GPU_IDS[@]}"; do
    GPU_ID=${GPU_IDS[$i]}
    START_INDEX=$(( i * LINES_PER_GPU ))
    END_INDEX=$(( (i + 1) * LINES_PER_GPU ))

    echo "Launching process for GPU ${GPU_ID} to process rows ${START_INDEX} to $(($END_INDEX - 1))"

    CUDA_VISIBLE_DEVICES=$GPU_ID python "$PYTHON_SCRIPT" \
        --csv_path "$CSV_PATH" \
        --save_dir "$SAVE_DIR" \
        --start-index "$START_INDEX" \
        --end-index "$END_INDEX" \
        "${PYTHON_ARGS[@]}" &

    PIDS+=($!)
done

echo "--------------------------------------------------"
echo "All processes launched. Waiting for completion..."
echo "Process IDs: ${PIDS[*]}"

EXIT_CODE=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "Error: Process with PID $PID failed."
        EXIT_CODE=1
    fi
done

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "--------------------------------------------------"
    echo "One or more tasks failed. Please check the logs."
    echo "--------------------------------------------------"
    exit 1
else
    echo "--------------------------------------------------"
    echo "All tasks completed successfully!"
    echo "Results are in: ${SAVE_DIR}"
    echo "--------------------------------------------------"
fi

exit 0
