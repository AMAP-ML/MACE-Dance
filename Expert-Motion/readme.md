# 🕺 Motion Expert

This folder contains the **Motion Expert** of **MACE-Dance**, which performs **music-to-3D dance motion generation**.

Given a music input, the Motion Expert generates a full-body **3D dance motion sequence** represented by **SMPL parameters**, aiming to produce motions that are both:

- **kinematically plausible**
- **artistically expressive**
- **well aligned with music rhythm**

As described in the paper, the Motion Expert adopts a **diffusion model** with a **BiMamba–Transformer hybrid architecture** and a **Guidance-Free Training (GFT)** strategy.

---

## 📁 Folder Structure

```bash
Expert-Motion/
├── args.py
├── test.py
├── EDGE.py
├── dataset/
├── model/
├── vis.py
├── data/
├── runs/
├── renders/
├── eval/
├── requirements.txt
└── readme.md
```

> The exact directory contents may vary slightly depending on checkpoints, cached datasets, and generated outputs.

---

## ⚠️ Important Note

This folder provides the **inference / evaluation pipeline** for the Motion Expert.

Please make sure that:

- the required dataset files are prepared correctly
- the checkpoint file exists
- the corresponding processed dataset cache is available if required

By default, `test.py` loads the checkpoint from:

```bash
./runs/train/exp/weights/train-3750.pt
```

If your checkpoint is stored elsewhere, please modify the path in `test.py` accordingly.

---

## 🧩 Environment Setup

We recommend creating a clean conda environment first:

```bash
conda create -n mace_motion python=3.10 -y
conda activate mace_motion
```

Install PyTorch according to your CUDA version. For example, with CUDA 12.1:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

In addition, please install **PyTorch3D** separately if it is not included in your environment:

```bash
pip install pytorch3d==0.7.8
```

> ⚠️ Please also make sure that `ffmpeg` is installed on your system, since it is required for audio-video rendering.

If you are using our internal environment, you may directly activate:

```bash
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega
```

---

## ▶️ Run Inference / Testing

Please follow the command below:

```bash
cd /mnt/workspace/yangkaixing/MACE-Dance/Expert-Motion
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=7 python test.py --batch_size=128 --feature_type baseline
```

---

## 📝 Arguments

The main testing script supports the following commonly used arguments:

- `--batch_size`  
  Batch size for inference/testing

- `--feature_type`  
  Type of music feature used by the model

In our default command, we use:

```bash
--batch_size=128 --feature_type baseline
```

---

## 📌 Notes

- `WANDB_MODE=offline` is recommended if you do not want to upload logs online.
- Please ensure that the checkpoint path in `test.py` is valid before running.
- Please ensure that the required dataset cache files exist under the expected directory.
- The Motion Expert outputs **3D motion**, not final rendered dance videos.
- Final visual dance videos are generated in the **Appearance Expert** stage.

---

## 📚 Reference

The Motion Expert corresponds to the **music-to-3D motion generation** stage in **MACE-Dance**, built on:

- a **Diffusion Model**
- a **BiMamba–Transformer hybrid architecture**
- **Guidance-Free Training (GFT)**

It serves as the first stage of the full music-driven dance video generation pipeline.
