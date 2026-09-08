# MACE-Dance Appearance Expert

Inference-only code for pose-driven character animation with
Wan2.2-Animate-14B, the MACE Body Adapter and the released Appearance LoRA.

The production path is intentionally fixed to:

```text
Wan2.2-Animate-14B + Body Adapter + 0.5 × Appearance LoRA
```

Only the Body Adapter's `pose_patch_embedding` tensors are loaded. Wan's
original face motion encoder, face encoder and face fusers remain untouched.
The LoRA strength is fixed to `0.5` in the Python entry point and is not exposed
as a command-line option.

## Environment

Python 3.10 and a recent CUDA-enabled PyTorch build are recommended.

```bash
conda create -n mace-appearance python=3.10 -y
conda activate mace-appearance
pip install -r requirements.txt
```

## Checkpoints

Download the Wan2.2-Animate-14B base model and the MACE-Dance Appearance Expert
weights, then arrange them as follows:

```text
Expert-Appearance/
└── models/
    ├── Wan-AI/Wan2.2-Animate-14B/
    │   ├── diffusion_pytorch_model-00001-of-00004.safetensors
    │   ├── diffusion_pytorch_model-00002-of-00004.safetensors
    │   ├── diffusion_pytorch_model-00003-of-00004.safetensors
    │   ├── diffusion_pytorch_model-00004-of-00004.safetensors
    │   ├── models_t5_umt5-xxl-enc-bf16.pth
    │   ├── Wan2.1_VAE.pth
    │   ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    │   └── google/umt5-xxl/
    └── train/
        ├── Adapter.safetensors
        └── LoRA.safetensors
```

- Base model: <https://huggingface.co/Wan-AI/Wan2.2-Animate-14B>
- Appearance Expert: <https://huggingface.co/GD-ML/MACE-Dance/tree/main/Expert-Appearance>

Custom locations can be supplied through `BASE_MODEL_DIR`, `ADAPTER_PATH` and
`LORA_PATH` when using the shell launcher, or through the corresponding Python
arguments.

## Input CSV

The launcher accepts a CSV with four required columns:

```csv
video,prompt,animate_pose_video,animate_face_video
data/0001/video.mp4,The character is dancing,data/0001/pose_video.mp4,data/0001/face_video.mp4
```

- `video`: reference image or video; the first frame is used as appearance input.
- `animate_pose_video`: full-resolution pose-control video.
- `animate_face_video`: face-control video; it is resized to `512 × 512`.
- `prompt`: text prompt for the generated video.
- `output_name`: optional output filename column.

Relative media paths are resolved against the directory containing the CSV.
The repository includes `test_data.csv` and seven small examples under `data/`.

## Inference

Run one process on one GPU:

```bash
cd Expert-Appearance
bash inference/infer.sh test_data.csv outputs/mace 0
```

Partition CSV rows over multiple GPUs:

```bash
bash inference/infer.sh test_data.csv outputs/mace 0,1,2,3
```

The multi-GPU launcher uses one independent inference process per GPU. It is
batch parallelism across CSV rows, not model parallelism for a single video.

Additional Python arguments can follow the first three launcher arguments:

```bash
bash inference/infer.sh test_data.csv outputs/mace 0,1 \
  --seed 0 --seed-per-row --overwrite
```

The maintained Python entry point can also be called directly:

```bash
python inference/appearance_infer.py \
  --csv_path test_data.csv \
  --save_dir outputs/mace \
  --adapter_path models/train/Adapter.safetensors \
  --lora_path models/train/LoRA.safetensors
```

## Fixed production defaults

```text
Body Adapter      enabled, pose projection only
Appearance LoRA   enabled, fixed alpha 0.5
height × width    1280 × 720
frames            81
sampling steps    20
CFG scale         1.0
seed              0
output FPS        15
```

Height and width must be divisible by 16. `num_frames` must be at least 5 and
satisfy `(num_frames - 1) % 4 == 0`. Wan Animate consumes `num_frames - 4`
pose/face control frames, so the default output contains 77 frames.

Resolution and sampling overrides are available as environment variables:

```bash
HEIGHT=1280 WIDTH=720 NUM_FRAMES=81 NUM_INFERENCE_STEPS=20 \
  bash inference/infer.sh test_data.csv outputs/mace 0
```

Existing videos are skipped by default. Pass `--overwrite` to regenerate them.
The launcher exits non-zero if any row or GPU shard fails.

## Notes

- Keep reference preprocessing, pose retargeting and face controls consistent
  across comparisons.
- Use height × width `1280 × 720`; this is `720 × 1280` in conventional width ×
  height notation.
- The first temporal block can contain a brief Wan VAE fade artifact.
  `--trim-start-frames` is available for previews but should not be used silently
  for quantitative evaluation.
