import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path = ["/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00001-of-00004.safetensors",
                            "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00002-of-00004.safetensors",
                            "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00003-of-00004.safetensors",
                            "/mnt/workspace/common/models/Wan2.2-Animate-14B/diffusion_pytorch_model-00004-of-00004.safetensors"]),
        ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/Wan2.1_VAE.pth"),
        ModelConfig(path="/mnt/workspace/common/models/Wan2.2-Animate-14B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
    tokenizer_config= ModelConfig(path = '/mnt/workspace/common/models/Wan2.2-Animate-14B/google/umt5-xxl/')
)
pipe.load_lora(pipe.dit, "/mnt/workspace/jiashu/DiffSynth-Studio/models/train/Wan2.2-Animate-14B_lora_480P_lorarank32_continue/step-1500.safetensors", alpha=1)
pipe.enable_vram_management()

input_image = VideoData("/mnt/workspace/yangkaixing/Reference/VideoGenData-ProcessAfterDownload/Final_Video_splited_results/7/zzkzmxy_zYTX_segment_000_clip_2/output.mp4", height=1280, width=720)[0]
animate_pose_video = VideoData("/mnt/workspace/yangkaixing/Reference/VideoGenData-ProcessAfterDownload/Final_Video_splited_results/7/zzkzmxy_zYTX_segment_000_clip_2/pose_video.mp4", height=1280, width=720).raw_data()[:81-4]
animate_face_video = VideoData("/mnt/workspace/yangkaixing/Reference/VideoGenData-ProcessAfterDownload/Final_Video_splited_results/7/zzkzmxy_zYTX_segment_000_clip_2/face_video.mp4", height=512, width=512).raw_data()[:81-4]
video = pipe(
    prompt="视频中的人在跳舞",
    seed=0, tiled=True,
    input_image=input_image,
    animate_pose_video=animate_pose_video,
    animate_face_video=animate_face_video,
    num_frames=81, height=1280, width=720,
    num_inference_steps=20, cfg_scale=1,
)
save_video(video, "video_Wan2.2-Animate-14B_480Plora.mp4", fps=15, quality=5)