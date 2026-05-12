#!/usr/bin/env bash
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
# num_processes=$((8 * NNODES))

torchrun --master_port=$MASTER_PORT --node_rank=$RANK --nnodes=$NNODES --master_addr=$MASTER_ADDR --nproc_per_node=8  \
    fastvideo/train_grpo_flux.py \
    --seed 42 \
    --pretrained_model_name_or_path /mnt/workspace/common/models/FLUX.1-dev \
    --vae_model_path /mnt/workspace/common/models/FLUX.1-dev \
    --cache_dir data/.cache \
    --data_json_path /mnt/workspace/jiashu/Research/DanceGRPO-main/data/rl_embeddings_flux_origin/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --num_latent_t 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 300 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 40 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_flux_pickscore_v1 \
    --h 720 \
    --w 720 \
    --t 1 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 0.1 \
    --weight_decay 0.0001 \
    --use_pickscore \
    --num_generations 12 \
    --shift 3 \
    --use_group \
    --ignore_last \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \