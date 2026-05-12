# 🎵 Motion Evaluation

This folder contains the **motion-dimension evaluation** pipeline for **music-driven dance video generation**.

As described in the paper, the motion evaluation focuses on whether generated dance motions are:

- 🕺 **Kinematic Fidelity**
- 🌈 **Diversity and Creativity**
- 🥁 **Well aligned with music beats**

To this end, we first extract **2D human keypoints** from videos using **ViTPose**, and then compute motion metrics from the **Human-Kinematics** perspective.

---

## 📊 Evaluation Metrics

We report the following motion metrics:

- **FID-k / FID-g**  
  Fréchet distance between generated motions and ground-truth motions in two feature spaces:
  - ⚡ **kinetic (k)**: captures motion dynamics
  - 📐 **geometric (g)**: captures spatial joint relationships

- **DIV-k / DIV-g**  
  Diversity of generated motions in the same two feature spaces:
  - ⚡ **kinetic (k)**
  - 📐 **geometric (g)**

- **BAS**  
  🥁 **Beat Alignment Score**, measuring synchronization between music and motion.

---

## 📁 Folder Structure

```bash
Evaluation-Motion/
├── Data/
│   ├── Pred/              # predicted/generated dance videos
│   ├── GT/                # ground-truth videos (and corresponding audio)
│   ├── Pred_ViTPose/      # extracted keypoints for predicted videos
│   └── GT_ViTPose/        # extracted keypoints for ground-truth videos
├── features/
├── calculate_metric.py    # metric computation script
├── extract_ViTPose.py     # keypoint extraction using ViTPose
├── metric.py
├── order.sh               # recommended execution order
├── readme.md
├── vitpose-s-coco_25.pth  # ViTPose checkpoint
└── yolov8s.pt             # detector checkpoint
```

---

## ⚠️ Important Note

The files currently placed in:

- `./Data/Pred`
- `./Data/GT`

are **only example/demo files selected for reference**.

When running the evaluation on your own results, you **must replace them** with:

- your own **Pred** videos in `./Data/Pred`
- the corresponding **GT** videos/audio in `./Data/GT`

> 🚨 In other words, please do **not** directly treat the provided `Pred` and `GT` files as the official evaluation set.

---

## 🧩 Environment Setup

We recommend creating a clean conda environment first:

```bash
conda create -n mace_motion_eval python=3.10 -y
conda activate mace_motion_eval
```

Install PyTorch according to your CUDA version. For example, with CUDA 12.1:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

In addition, this project depends on **easy-ViTPose**.  
Please install it separately from your local source:

```bash
pip install -e /path/to/easy_ViTPose
```

> If you are using our internal environment, you may also directly activate:
>
> ```bash
> conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega
> ```

---

## ▶️ Run the Evaluation

Please follow the execution order below. You can also refer to `order.sh`.

### 1️⃣ Extract ViTPose keypoints for prediction videos

```bash
CUDA_VISIBLE_DEVICES=7 python extract_ViTPose.py \
  --video_dir ./Data/Pred \
  --output_dir ./Data/Pred_ViTPose \
  --single_pose --is_video
```

### 2️⃣ Extract ViTPose keypoints for ground-truth videos

```bash
CUDA_VISIBLE_DEVICES=7 python extract_ViTPose.py \
  --video_dir ./Data/GT \
  --output_dir ./Data/GT_ViTPose \
  --single_pose --is_video
```

### 3️⃣ Calculate motion metrics

```bash
python calculate_metric.py \
  --gt_path ./Data/GT_ViTPose \
  --pred_path ./Data/Pred_ViTPose \
  --audio_path ./Data/GT
```

---

## 📝 Input Requirements

### 🎬 1. Prediction videos

Place your generated dance videos in:

```bash
./Data/Pred
```

### 🎼 2. Ground-truth videos / audio

Place the corresponding ground-truth data in:

```bash
./Data/GT
```

The evaluation script uses:

- ✅ **GT keypoints** for motion fidelity comparison
- ✅ **audio from GT** for beat alignment evaluation

So the files in `Pred` and `GT` should be properly matched.

---

## 📦 Output

After running the pipeline:

- extracted keypoints for predicted videos will be saved to:

```bash
./Data/Pred_ViTPose
```

- extracted keypoints for ground-truth videos will be saved to:

```bash
./Data/GT_ViTPose
```

- the final metric results will be printed by `calculate_metric.py`

Typical reported metrics include:

- `FID-k`
- `FID-g`
- `DIV-k`
- `DIV-g`
- `BAS`

---

## ✅ Recommended Usage

For a fair evaluation:

1. Replace the demo files in `Data/Pred` and `Data/GT` with your own data.
2. Ensure prediction videos and GT videos/audio correspond correctly.
3. Run the scripts in the exact order shown above.
4. Use the same conda environment and checkpoints provided in this folder.

---

## 📚 Reference

This evaluation protocol follows the motion-dimension design in our paper:

- 🧍 2D pose extraction with **ViTPose**
- ⚡ motion fidelity/diversity in **kinetic** and **geometric** feature spaces
- 🥁 music-motion synchronization using **Beat Alignment Score (BAS)**
