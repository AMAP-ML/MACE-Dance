# 🎭 Expert-Appearance

> **Appearance Expert of MACE-Dance**  
> Pose-driven image animation for music-driven dance video generation.

---

## 📌 Overview

This repository contains the **Appearance Expert** of **MACE-Dance**, a cascaded framework for **music-driven dance video generation**.

In the full **MACE-Dance** pipeline:

- **Motion Expert** generates music-aligned **3D dance motion**
- **Appearance Expert** converts motion / pose conditions into **high-quality dancing videos**

This repository focuses on the **Appearance Expert**, which takes a reference image and a driving pose sequence, then synthesizes a temporally coherent dance video while preserving subject identity and visual fidelity.

---

## 🌟 Features

- 🎬 Pose-driven dance animation
- 🧍 Identity-preserving generation
- ⏱️ Temporal consistency across frames
- 🎨 Dance-specific visual adaptation
- 🧩 Easy integration into the full **MACE-Dance** pipeline

---

## 🧠 Method Summary

The **Appearance Expert** in **MACE-Dance** is trained with a **Kinematic–Aesthetic decoupled fine-tuning strategy**.

### 1. Kinematic Stage

- Fine-tunes the **Body Adapter**
- Strengthens motion / pose adherence
- Freezes the **DiT backbone** and **VAE**

### 2. Aesthetic Stage

- Freezes the kinematic pathway
- Fine-tunes extended **LoRA** branches
- Improves texture fidelity, identity consistency, and temporal coherence

This decoupled design helps better balance:

- motion controllability
- visual realism
- subject consistency
- long-range temporal stability

---

## 🗂️ Project Structure

```bash
Expert-Appearance/
├── data/                  # All data should be prepared under this folder
├── diffsynth/             # Core diffusion / synthesis modules
├── examples/              # Example files and demos
├── inference/             # Inference scripts
├── models/                # Checkpoints and trained weights
├── outputs/               # Generated results
├── readme.md              # Documentation
├── requirements.txt       # Dependency list
└── test_data.csv          # Inference metadata file
```

---

## 🔧 Environment Setup

Please make sure the required dependencies are installed and the target conda environment is available.

```bash
conda activate /mnt/workspace/jiashu/anaconda3/envs/cogvideo_clone
```

If needed, install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📦 Data

All data should be prepared under:

```bash
./data
```

The data organization follows the original **WAN-Animate** setting.

For detailed data preparation instructions, please refer to the original **WAN-Animate** paper or repository.

---

## 🚀 Inference

Run the following commands for inference:

```bash
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
```

---

## 📄 Input Specification

The inference script uses the following files:

- `./test_data.csv` as the input metadata file
- `./models/train/LoRA.safetensors` as the LoRA checkpoint
- `./models/train/Adapter.safetensors` as the adapter checkpoint

Please make sure all referenced files exist before running inference.

---

## 📝 Notes

- **Module Scope**: This repository contains the **Appearance Expert** only, designed to integrate with the full **MACE-Dance** pipeline.
- **Backbone**: The **DiT (Diffusion Transformer) backbone** follows the original **WAN-Animate** implementation.
- **Performance Disclaimer**:
    - We would like to be transparent about our contribution: the performance gains of this **Appearance Expert** over the original **WAN-Animate** are relatively modest. 
    - In certain aspects or specific evaluation metrics, the original **WAN-Animate** may still demonstrate superior performance.
    - If you are conducting research in this field, we strongly recommend using **WAN-Animate** as your primary baseline for further exploration and development.
- **Data Protocol**: Data preparation follows the **WAN-Animate** standard.
- **Model Checkpoints**: Please ensure model checkpoints are placed in the correct directory before inference.

---

## 📚 Relation to the Paper

This repository implements the **Appearance Expert** described in:

> **MACE-Dance: Motion-Appearance Cascaded Experts for Music-Driven Dance Video Generation**

The Appearance Expert is designed to produce dance videos with:

- high perceptual quality
- strong temporal consistency
- robust identity preservation

---

## ⭐ Final Words

If this project helps your research or production, that is wonderful.  
May your models dance smoothly and your videos stay consistent. 🕺🎶
