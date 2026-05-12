# 1*80G GPU cannot train Wan2.2-Animate-14B LoRA
# We tested on 8*80G GPUs

export PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/bin:$PATH"
export LIBRARY_PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/workspace/yangcundian/cache/envs/cuda-12.4/lib64:$LD_LIBRARY_PATH"
export CUDA_HOME="/mnt/workspace/yangcundian/cache/envs/cuda-12.4"


accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path '' \
  --dataset_metadata_path /mnt/workspace/jiashu/DiffSynth-Studio/video_data.csv \
  --data_file_keys "video,animate_pose_video,animate_face_video" \
  --height 1280 \
  --width 720 \
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
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/mnt/workspace/jiashu/DiffSynth-Studio/models/train/Wan2.2-Animate-14B_lora_720P_lorarank32" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload