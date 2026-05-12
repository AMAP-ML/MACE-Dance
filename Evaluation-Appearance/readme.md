# 🎨 Appearance Evaluation

This folder contains the **appearance-dimension evaluation** pipeline for **music-driven dance video generation**.

We adopt **VBench** as the core benchmark for video appearance evaluation, and select a set of dance-related metrics to assess visual quality and temporal consistency.

The evaluated metrics include:

- **IQ**: Imaging Quality
- **AQ**: Aesthetic Quality
- **SC**: Subject Consistency
- **BC**: Background Consistency
- **MS**: Motion Smoothness
- **TF**: Temporal Flickering

---

## 📁 Folder Structure

```bash
Evaluation-Appearance/
├── Pred/                         # input videos to be evaluated
├── Pred_Out/                     # output evaluation results
├── VBench/                       # local VBench codebase
├── eval_all_dimension_multi.sh   # evaluation script
├── order.sh                      # recommended execution order
├── requirements.txt
└── readme.md
```

---

## ⚠️ Important Note

The videos currently placed in:

```bash
./Pred
```

are **only example/demo files**.

When running the evaluation on your own results, please **replace them with your own generated videos**.

---

## 🧩 Environment Setup

We recommend creating a dedicated conda environment first:

```bash
conda create -n mace_appearance_eval python=3.8 -y
conda activate mace_appearance_eval
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

If you are using our internal environment, you may also directly activate:

```bash
conda activate /mnt/workspace/yangkaixing/CONDA_ENV/mega
```

However, the actual VBench evaluation in our script uses the following environment:

```bash
conda activate /mnt/workspace/lingxinran/miniconda3/envs/vbench
```

Please make sure that the VBench-related dependencies are correctly installed in that environment.

---

## 📦 VBench and Pretrained Weights

This evaluation is based on **VBench**, which requires several pretrained models.

### Option 1: Use our local VBench folder
You may directly use the `./VBench` folder provided here.

### Option 2: Use the official VBench repository
We also recommend directly using the **official VBench repository**, which may be more convenient for dependency setup and pretrained model preparation.  
In practice, using the official VBench repository is also perfectly fine and yields the **same evaluation results** for our selected dimensions.

Official repository:

```text
https://github.com/Vchitect/VBench
```

### Pretrained weights
Since VBench depends on multiple pretrained checkpoints, please follow the official VBench instructions to download the required weights.

After downloading, place the pretrained files under your VBench pretrained directory, for example:

```bash
VBench/pretrained/
```

or link it to your local pretrained cache directory.

> ⚠️ Note: If the pretrained weights are missing, VBench may try to download them automatically during evaluation, which can be very slow depending on your network environment.

---

## ▶️ Run the Evaluation

Please follow the execution order below. You can also refer to `order.sh`.

### Step 1: Go to the evaluation folder

```bash
cd /mnt/workspace/yangkaixing/MACE-Dance/Evaluation-Appearance
```

### Step 2: Run the evaluation script

```bash
bash eval_all_dimension_multi.sh \
  --videos_path ../Pred \
  --output_path ../Pred_Out
```

---

## 📝 Input and Output

### Input
Place the videos to be evaluated in:

```bash
./Pred
```

### Output
The evaluation results will be saved to:

```bash
./Pred_Out
```

---

## 📌 Notes

- Please replace the example videos in `Pred` with your own videos before evaluation.
- Some VBench dimensions, especially **motion_smoothness**, can be memory-intensive for high-resolution videos.
- If CUDA OOM occurs, we recommend resizing the videos before evaluation.

---

## ✅ Recommended Usage

For a smooth evaluation process:

1. Prepare your own generated videos in `./Pred`
2. Install the required environment
3. Prepare the VBench pretrained weights
4. Run `eval_all_dimension_multi.sh`
5. Check the results in `./Pred_Out`

---

## 📚 Reference

This appearance evaluation follows the **VBench-based appearance protocol** used in our project for dance video generation.
