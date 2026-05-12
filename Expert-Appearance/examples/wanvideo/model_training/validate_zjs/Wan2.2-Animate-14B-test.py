import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def get_project_root():
    """
    根据当前脚本路径反推项目根目录。
    假设当前文件位于:
    Expert-Appearance/examples/wanvideo/model_training/validate_zjs/Wan2.2-Animate-14B-test.py

    那么 project_root = 往上 4 层
    """
    return Path(__file__).resolve().parents[4]


def build_default_paths(project_root: Path):
    return {
        "base_model_dir": project_root / "models" / "Wan-AI" / "Wan2.2-Animate-14B",
        "adapter_path": project_root / "models" / "train" / "Adapter.safetensors",
        "default_lora_path": project_root / "models" / "train" / "LoRA.safetensors",
    }


def create_pipeline(base_model_dir, adapter_path=None, lora_path=None, lora_alpha=1.0):
    print("Loading WanVideoPipeline model...")
    print(f"Base model dir: {base_model_dir}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=[
                str(base_model_dir / "diffusion_pytorch_model-00001-of-00004.safetensors"),
                str(base_model_dir / "diffusion_pytorch_model-00002-of-00004.safetensors"),
                str(base_model_dir / "diffusion_pytorch_model-00003-of-00004.safetensors"),
                str(base_model_dir / "diffusion_pytorch_model-00004-of-00004.safetensors"),
            ]),
            ModelConfig(path=str(base_model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(base_model_dir / "Wan2.1_VAE.pth")),
            ModelConfig(path=str(base_model_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(base_model_dir / "google" / "umt5-xxl"))
    )

    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        if adapter_path.exists():
            print(f"Loading animate adapter from: {adapter_path}")
            state_dict = load_state_dict(str(adapter_path))
            missing, unexpected = pipe.animate_adapter.load_state_dict(state_dict, strict=False)
            print(f"Animate adapter loaded. missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"[WARN] Adapter not found: {adapter_path}")

    if lora_path is not None:
        lora_path = Path(lora_path)
        if lora_path.exists():
            print(f"Loading LoRA from: {lora_path} with alpha={lora_alpha}")
            pipe.load_lora(pipe.dit, str(lora_path), alpha=lora_alpha)
        else:
            print(f"[WARN] LoRA not found: {lora_path}")

    pipe.enable_vram_management()
    print("Model loaded successfully.")
    return pipe


def main():
    parser = argparse.ArgumentParser(description="Batch inference script for WanVideoPipeline.")

    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated videos.")

    parser.add_argument("--base_model_dir", type=str, default=None, help="Base Wan2.2-Animate-14B model directory.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Animate adapter path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA file.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha value for LoRA.")

    parser.add_argument("--start-index", type=int, default=0, help="Start row index (inclusive) of the CSV to process.")
    parser.add_argument("--end-index", type=int, default=None, help="End row index (exclusive) of the CSV to process.")

    parser.add_argument("--height", type=int, default=1280, help="Height of the output video.")
    parser.add_argument("--width", type=int, default=720, help="Width of the output video.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale.")

    parser.add_argument("--fps", type=int, default=15, help="FPS for the saved video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality for the saved video.")

    args = parser.parse_args()

    project_root = get_project_root()
    defaults = build_default_paths(project_root)

    base_model_dir = Path(args.base_model_dir) if args.base_model_dir else defaults["base_model_dir"]
    adapter_path = Path(args.adapter_path) if args.adapter_path else defaults["adapter_path"]
    lora_path = Path(args.lora_path) if args.lora_path else None

    print(f"Project root: {project_root}")
    print(f"Using base_model_dir: {base_model_dir}")
    print(f"Using adapter_path: {adapter_path}")
    print(f"Using lora_path: {lora_path}")

    pipe = create_pipeline(
        base_model_dir=base_model_dir,
        adapter_path=adapter_path,
        lora_path=lora_path,
        lora_alpha=args.lora_alpha,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    start_index = args.start_index
    end_index = args.end_index if args.end_index is not None else len(df)
    df_subset = df.iloc[start_index:end_index]

    if df_subset.empty:
        print(f"Warning: No data to process for index range {start_index} to {end_index}. Exiting.")
        return

    print(f"Starting batch processing for rows {start_index} to {end_index - 1}...")

    for index, row in tqdm(
        df_subset.iterrows(),
        total=len(df_subset),
        desc=f"Processing rows {start_index}-{end_index-1}"
    ):
        video_path = row["video"]
        prompt = row["prompt"]
        pose_video_path = row["animate_pose_video"]
        face_video_path = row["animate_face_video"]

        base_name = video_path.split("/")[-2] + ".mp4"
        output_path = os.path.join(args.save_dir, base_name)

        if os.path.exists(output_path):
            print(f"Warning: Output file {output_path} already exists. Skipping.")
            continue

        try:
            input_image = VideoData(video_path, height=args.height, width=args.width)[0]
            animate_pose_video = VideoData(pose_video_path, height=1280, width=720).raw_data()[:args.num_frames - 4]
            animate_face_video = VideoData(face_video_path, height=512, width=512).raw_data()[:args.num_frames - 4]

            video = pipe(
                prompt=prompt,
                seed=args.seed,
                tiled=True,
                input_image=input_image,
                animate_pose_video=animate_pose_video,
                animate_face_video=animate_face_video,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )

            save_video(video, output_path, fps=args.fps, quality=args.quality)

        except Exception as e:
            print(f"\n[ERROR] Failed to process item at row {index} ({base_name}). Error: {e}")
            continue


if __name__ == "__main__":
    main()
