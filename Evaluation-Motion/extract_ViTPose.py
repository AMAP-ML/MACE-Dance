import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from easy_ViTPose import VitInference
from easy_ViTPose.vit_utils.inference import NumpyEncoder
from easy_ViTPose.vit_utils.visualization import joints_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Extract pose keypoints from videos using easy_ViTPose.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output JSON files.")
    parser.add_argument("--model_path", type=str, default="./vitpose-s-coco_25.pth", help="Path to ViTPose model.")
    parser.add_argument("--yolo_path", type=str, default="./yolov8s.pt", help="Path to YOLO model.")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. cuda:0 or cpu.")
    parser.add_argument("--model_name", type=str, default="s", help="ViTPose model size, e.g. s/b/l.")
    parser.add_argument("--yolo_size", type=int, default=320, help="YOLO input size.")
    parser.add_argument("--single_pose", action="store_true", help="Use single pose mode.")
    parser.add_argument("--is_video", action="store_true", help="Enable video tracking mode.")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = VitInference(
        args.model_path,
        args.yolo_path,
        model_name=args.model_name,
        yolo_size=args.yolo_size,
        is_video=args.is_video,
        single_pose=args.single_pose,
        device=args.device
    )

    valid_exts = {".mp4", ".avi", ".mov", ".mkv"}
    files = sorted([
        f for f in os.listdir(args.video_dir)
        if os.path.isfile(os.path.join(args.video_dir, f)) and os.path.splitext(f)[1].lower() in valid_exts
    ])

    print(f"Found {len(files)} video files in {args.video_dir}")

    for file in tqdm(files, desc="Processing videos"):
        video_path = os.path.join(args.video_dir, file)
        output_json_path = os.path.join(args.output_dir, os.path.splitext(file)[0] + ".json")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_keypoints = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            keypoints_dict = model.inference(rgb_frame)
            json_ready = {pid: kpt.tolist() for pid, kpt in keypoints_dict.items()}

            if 0 in json_ready:
                all_keypoints.append(json_ready[0])
            else:
                all_keypoints.append([[0, 0, 0]] * 25)

            frame_idx += 1

        cap.release()

        output_data = {
            "video_name": os.path.basename(video_path),
            "fps": fps,
            "total_frames": frame_idx,
            "skeleton": joints_dict()[model.dataset]["keypoints"],
            "keypoints": all_keypoints
        }

        with open(output_json_path, "w") as f:
            json.dump(output_data, f, cls=NumpyEncoder, indent=2)

        all_keypoints_np = np.array(all_keypoints)
        print(f"{file}: keypoints shape = {all_keypoints_np.shape}")

        model.reset()

        print(f"✅ Keypoint JSON saved to: {output_json_path}")


if __name__ == "__main__":
    main()
