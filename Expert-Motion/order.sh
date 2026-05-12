cd /mnt/workspace/yangkaixing/MACE-Dance/Expert-Motion
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=7 python test.py --batch_size=128 --feature_type baseline