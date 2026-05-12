cd /mnt/workspace/yangkaixing/MACE-Dance/Evaluation-Motion
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega

CUDA_VISIBLE_DEVICES=7 python extract_ViTPose.py \
  --video_dir ./Data/Pred \
  --output_dir ./Data/Pred_ViTPose \
  --single_pose --is_video

CUDA_VISIBLE_DEVICES=7 python extract_ViTPose.py \
  --video_dir ./Data/GT \
  --output_dir ./Data/GT_ViTPose \
  --single_pose --is_video

python calculate_metric.py \
  --gt_path ./Data/GT_ViTPose \
  --pred_path ./Data/Pred_ViTPose \
  --audio_path ./Data/GT


