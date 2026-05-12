cd /mnt/workspace/yangkaixing/MACE-Dance/Evaluation-Appearance
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega

CUDA_VISIBLE_DEVICES=7 sh eval_all_dimension_multi.sh \
  --videos_path ../Pred \
  --output_path ../Pred_Out
