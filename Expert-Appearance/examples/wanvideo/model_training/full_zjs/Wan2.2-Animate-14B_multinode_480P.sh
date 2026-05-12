# 1*80G GPU cannot train Wan2.2-Animate-14B LoRA
# We tested on 8*80G GPUs

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=1200000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
export GPUS_PER_NODE=8
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
export HF_ENDPOINT=https://hf-mirror.com
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
# WORLD_SIZE=$1
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "NNODES:$NNODES"
echo "MASTER_ADDR:$MASTER_ADDR"
echo "NUM_PROCESSES:$NUM_PROCESSES"
WORLD_SIZE=$NUM_PROCESSES

export PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/bin:$PATH"
export LIBRARY_PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/mnt/workspace/yangcundian/cache/envs/cuda-12.4"

accelerate launch --config_file /mnt/workspace/jiashu/DiffSynth-Studio/examples/wanvideo/model_training/lora_zjs/accelerate_config_14B_multinode.yaml \
  --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT \
  --machine_rank=$NODE_RANK --num_processes=$NUM_PROCESSES --num_machines=$NNODES \
    examples/wanvideo/model_training/train.py \
  --dataset_base_path '' \
  --dataset_metadata_path /mnt/workspace/jiashu/DiffSynth-Studio/video_data.csv \
  --data_file_keys "video,animate_pose_video,animate_face_video" \
  --height 832 \
  --width 480 \
  --num_frames 81 \
  --dataset_repeat 1 \
  --model_paths '[
    [
      "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00001-of-00004.safetensors",
      "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00002-of-00004.safetensors",
      "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00003-of-00004.safetensors",
      "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00004-of-00004.safetensors"
    ],
    "/mnt/workspace/common/models/Wan2.2-Animate-14B/models_t5_umt5-xxl-enc-bf16.pth",
    "/mnt/workspace/common/models/Wan2.2-Animate-14B/Wan2.1_VAE.pth",
    "/mnt/workspace/common/models/Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
  ]' \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.animate_adapter." \
  --output_path "./models/train/Wan2.2-Animate-14B_full_animate_adapter_480P" \
  --trainable_models "animate_adapter" \
  --extra_inputs "input_image,animate_pose_video,animate_face_video" \
  --use_gradient_checkpointing_offload