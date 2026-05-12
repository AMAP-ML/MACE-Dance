#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        --videos_path)
            VIDEOS_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_vbench.sh --videos_path <path> --output_path <path>"
            exit 1
            ;;
    esac
done

if [ -z "$VIDEOS_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: bash run_vbench.sh --videos_path <path> --output_path <path>"
    exit 1
fi

eval "$(/opt/conda/condabin/conda shell.bash hook)"
conda activate /mnt/workspace/lingxinran/miniconda3/envs/vbench

cd VBench || exit 1

export HF_HOME=/mnt/workspace/zhuchen/cache
export HF_ENDPOINT=https://hf-mirror.com
# export VBENCH_CACHE_DIR=./VBench/pretrained

export NCCL_DEBUG=WARN
export MASTER_PORT=17211

python evaluate.py \
    --dimension "imaging_quality" "aesthetic_quality" "subject_consistency" "background_consistency" "motion_smoothness" "temporal_flickering" \
    --videos_path "${VIDEOS_PATH}" \
    --load_ckpt_from_local True \
    --mode=custom_input \
    --output_path "${OUTPUT_PATH}"
