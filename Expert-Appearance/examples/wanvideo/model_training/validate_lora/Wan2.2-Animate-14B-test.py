import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import argparse
import pandas as pd
import os
from tqdm import tqdm

def main():
    # 1. 使用 argparse 设置可传入的参数
    parser = argparse.ArgumentParser(description="Batch inference script for WanVideoPipeline.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated videos.")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional path to a LoRA file.")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="Alpha value for LoRA.")
    
    # 推理参数
    parser.add_argument("--height", type=int, default=1280, help="Height of the output video.")
    parser.add_argument("--width", type=int, default=720, help="Width of the output video.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale.")
    
    # 视频保存参数
    parser.add_argument("--fps", type=int, default=15, help="FPS for the saved video.")
    parser.add_argument("--quality", type=int, default=5, help="Quality for the saved video.")

    args = parser.parse_args()

    # --- 模型加载 ---
    print("Loading WanVideoPipeline model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=["/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00001-of-00004.safetensors",
                                "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00002-of-00004.safetensors",
                                "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00003-of-00004.safetensors",
                                "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00004-of-00004.safetensors"]),
            ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/Wan2.1_VAE.pth"),
            ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(path='/mnt/workspace/common/models/Wan2.2-Animate-14B/google/umt5-xxl/')
    )

    # 如果提供了LoRA路径，则加载LoRA
    if args.lora_path:
        print(f"Loading LoRA from: {args.lora_path} with alpha={args.lora_alpha}")
        pipe.load_lora(pipe.dit, args.lora_path, alpha=args.lora_alpha)
        
    pipe.enable_vram_management()
    print("Model loaded successfully.")

    # 3. 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. 读取CSV文件
    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    # 遍历CSV中的每一行进行处理
    print(f"Starting batch processing for {len(df)} items...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
        video_path = row['video']
        prompt = row['prompt']
        pose_video_path = row['animate_pose_video']
        face_video_path = row['animate_face_video']

        # 为保存文件生成一个唯一的名字（使用原始视频的文件名）
        base_name = os.path.basename(video_path)
        output_path = os.path.join(args.save_dir, base_name)

        print(f"\nProcessing item {index+1}/{len(df)}: {base_name}")
        
        try:
            # 准备输入数据
            # input_image 为视频的第一帧
            input_image = VideoData(video_path, height=args.height, width=args.width)[0]
            
            # 这里的切片逻辑 `[:args.num_frames - 4]` 是根据您原始代码保留的，确保控制视频长度与生成帧数匹配
            # 注意: 这里的控制视频分辨率是硬编码的，如果您的数据分辨率不同，可能需要调整
            animate_pose_video = VideoData(pose_video_path, height=1280, width=720).raw_data()[:args.num_frames-4]
            animate_face_video = VideoData(face_video_path, height=512, width=512).raw_data()[:args.num_frames-4]

            # 执行推理
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
            
            # 保存视频
            save_video(video, output_path, fps=args.fps, quality=args.quality)
            print(f"Successfully generated and saved to: {output_path}")

        except Exception as e:
            print(f"Failed to process {base_name}. Error: {e}")
            # 您可以选择在这里记录错误日志或跳过
            continue

if __name__ == "__main__":
    main()
