cd /mnt/workspace/yangkaixing/MACE-Dance/Expert-Apprearance
conda activate /mnt/workspace/jiashu/anaconda3/envs/cogvideo_clone
export PYTHONPATH=/mnt/workspace/yangkaixing/MACE-Dance/Expert-Apprearance

sh ./inference/test_siggraph.sh \
  --csv-path ./test_data.csv \
  --save-dir ./outputs \
  --gpus "1,2,3,4" \
  -- \
  --lora_path ./models/train/LoRA.safetensors \
  --adapter_path ./models/train/Adapter.safetensors