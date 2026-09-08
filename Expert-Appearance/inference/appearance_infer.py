"""Portable single-GPU batch inference for the MACE Appearance Expert."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from diffsynth import VideoData, load_state_dict, save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
from diffsynth.utils.animate_adapter import (
    animate_adapter_component_counts,
    select_body_adapter_state_dict,
)


REQUIRED_COLUMNS = (
    "video",
    "prompt",
    "animate_pose_video",
    "animate_face_video",
)
PATH_COLUMNS = ("video", "animate_pose_video", "animate_face_video")
LORA_ALPHA = 0.5


def project_root() -> Path:
    return PROJECT_ROOT


def default_paths(root: Path) -> dict[str, Path]:
    return {
        "base_model_dir": root / "models" / "Wan-AI" / "Wan2.2-Animate-14B",
        "adapter_path": root / "models" / "train" / "Adapter.safetensors",
        "lora_path": root / "models" / "train" / "LoRA.safetensors",
    }


def resolve_csv_paths(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Resolve relative media paths against the CSV directory."""
    frame = frame.copy()
    csv_dir = csv_path.resolve().parent
    for column in PATH_COLUMNS:
        frame[column] = frame[column].map(
            lambda value: str(
                Path(str(value))
                if Path(str(value)).is_absolute()
                else (csv_dir / str(value)).resolve()
            )
        )
    return frame


def validate_base_model(base_model_dir: Path) -> None:
    required = [
        *(
            base_model_dir
            / f"diffusion_pytorch_model-{part:05d}-of-00004.safetensors"
            for part in range(1, 5)
        ),
        base_model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        base_model_dir / "Wan2.1_VAE.pth",
        base_model_dir
        / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        base_model_dir / "google" / "umt5-xxl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing base-model files:\n  " + "\n  ".join(missing))


def create_pipeline(
    base_model_dir: Path,
    adapter_path: Path,
    lora_path: Path,
):
    validate_base_model(base_model_dir)
    print(f"Loading Wan-Animate base model from {base_model_dir}", flush=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                path=[
                    str(
                        base_model_dir
                        / "diffusion_pytorch_model-00001-of-00004.safetensors"
                    ),
                    str(
                        base_model_dir
                        / "diffusion_pytorch_model-00002-of-00004.safetensors"
                    ),
                    str(
                        base_model_dir
                        / "diffusion_pytorch_model-00003-of-00004.safetensors"
                    ),
                    str(
                        base_model_dir
                        / "diffusion_pytorch_model-00004-of-00004.safetensors"
                    ),
                ]
            ),
            ModelConfig(
                path=str(base_model_dir / "models_t5_umt5-xxl-enc-bf16.pth")
            ),
            ModelConfig(path=str(base_model_dir / "Wan2.1_VAE.pth")),
            ModelConfig(
                path=str(
                    base_model_dir
                    / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
                )
            ),
        ],
        tokenizer_config=ModelConfig(
            path=str(base_model_dir / "google" / "umt5-xxl")
        ),
    )

    if not adapter_path.is_file():
        raise FileNotFoundError(f"Body Adapter checkpoint not found: {adapter_path}")
    checkpoint = load_state_dict(str(adapter_path))
    selected = select_body_adapter_state_dict(checkpoint)
    missing, unexpected = pipe.animate_adapter.load_state_dict(selected, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Body Adapter keys: {unexpected}")
    print(
        "Loaded Body Adapter "
        f"components={animate_adapter_component_counts(selected)}, "
        f"Wan face components retained={len(missing)}",
        flush=True,
    )

    if not lora_path.is_file():
        raise FileNotFoundError(f"Appearance LoRA checkpoint not found: {lora_path}")
    print(f"Loading Appearance LoRA from {lora_path} with fixed alpha={LORA_ALPHA}")
    pipe.load_lora(pipe.dit, str(lora_path), alpha=LORA_ALPHA)

    pipe.enable_vram_management()
    return pipe


def load_controls(row, height: int, width: int, control_frames: int):
    input_image = VideoData(str(row["video"]), height=height, width=width)[0]
    pose = VideoData(
        str(row["animate_pose_video"]), height=height, width=width
    ).raw_data()
    face = VideoData(
        str(row["animate_face_video"]), height=512, width=512
    ).raw_data()
    if len(pose) < control_frames or len(face) < control_frames:
        raise ValueError(
            "Control video is too short: "
            f"need {control_frames} frames, got pose={len(pose)}, face={len(face)}"
        )
    return input_image, pose[:control_frames], face[:control_frames]


def output_name(row, index: int) -> str:
    if "output_name" in row and pd.notna(row["output_name"]):
        name = Path(str(row["output_name"])).name
        return name if name.endswith(".mp4") else f"{name}.mp4"
    source = Path(str(row["video"]))
    stem = source.parent.name or source.stem or f"row-{index:06d}"
    return f"{stem}.mp4"


def parse_args():
    defaults = default_paths(project_root())
    parser = argparse.ArgumentParser(
        description="MACE-Dance Appearance Expert single-GPU batch inference"
    )
    parser.add_argument("--csv_path", "--csv-path", required=True)
    parser.add_argument("--save_dir", "--save-dir", required=True)
    parser.add_argument(
        "--base_model_dir",
        "--base-model-dir",
        default=str(defaults["base_model_dir"]),
    )
    parser.add_argument(
        "--adapter_path",
        "--adapter-path",
        default=str(defaults["adapter_path"]),
    )
    parser.add_argument(
        "--lora_path",
        "--lora-path",
        default=str(defaults["lora_path"]),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", "--num-frames", type=int, default=81)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-per-row", action="store_true")
    parser.add_argument(
        "--num_inference_steps",
        "--num-inference-steps",
        type=int,
        default=20,
    )
    parser.add_argument("--cfg_scale", "--cfg-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--trim-start-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must both be divisible by 16")
    if args.num_frames < 5 or (args.num_frames - 1) % 4:
        raise ValueError(
            "num_frames must be at least 5 and satisfy (num_frames - 1) % 4 == 0"
        )
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("end-index must not be smaller than start-index")
    if args.trim_start_frames < 0:
        raise ValueError("trim-start-frames must be non-negative")


def main():
    args = parse_args()
    validate_args(args)
    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    frame = pd.read_csv(csv_path)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in frame]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")
    missing_values = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if missing_values.any():
        rows = frame.index[missing_values].tolist()
        raise ValueError(f"CSV contains missing required values in rows: {rows}")
    frame = resolve_csv_paths(frame, csv_path)

    end_index = (
        len(frame) if args.end_index is None else min(args.end_index, len(frame))
    )
    subset = frame.iloc[args.start_index : end_index]
    if subset.empty:
        print(f"No rows in range [{args.start_index}, {end_index}); nothing to do.")
        return

    pipe = create_pipeline(
        Path(args.base_model_dir),
        Path(args.adapter_path),
        Path(args.lora_path),
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    control_frames = args.num_frames - 4
    failures = []
    for index, row in tqdm(
        subset.iterrows(), total=len(subset), desc="Appearance inference"
    ):
        path = save_dir / output_name(row, int(index))
        if path.exists() and not args.overwrite:
            print(f"Skipping existing output: {path}")
            continue
        try:
            input_image, pose, face = load_controls(
                row, args.height, args.width, control_frames
            )
            seed = args.seed + int(index) if args.seed_per_row else args.seed
            video = pipe(
                prompt=str(row["prompt"]),
                seed=seed,
                tiled=True,
                input_image=input_image,
                animate_pose_video=pose,
                animate_face_video=face,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
            )
            if args.trim_start_frames:
                video = video[args.trim_start_frames :]
                if len(video) == 0:
                    raise ValueError("trim-start-frames removed the entire result")
            save_video(video, str(path), fps=args.fps, quality=args.quality)
            print(f"Saved row {index}: {path}", flush=True)
        except Exception as exc:
            failures.append((int(index), str(exc)))
            print(f"[ERROR] row {index}: {exc}", flush=True)

    if failures:
        details = "; ".join(
            f"row {index}: {message}" for index, message in failures
        )
        raise RuntimeError(f"{len(failures)} inference row(s) failed: {details}")


if __name__ == "__main__":
    main()
