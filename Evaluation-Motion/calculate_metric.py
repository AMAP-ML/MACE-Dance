import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import librosa

from features.geometric_2d import geometric_features
from features.kinetic_2d import kinetic_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dance motion metrics from keypoint JSON files.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to GT JSON folder.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to prediction JSON folder.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to GT audio/video folder for beat extraction.")
    parser.add_argument("--json_key", type=str, default="keypoints", help="Key name for keypoints in JSON.")
    parser.add_argument("--fixed_length", type=int, default=64, help="Fixed number of frames to keep.")
    parser.add_argument("--audio_sr", type=int, default=8320, help="Audio sample rate for beat extraction.")
    parser.add_argument("--fps", type=int, default=16, help="Video FPS for converting beat time to frame index.")
    return parser.parse_args()


def normalize(gt, pred):
    gt_mean = gt.mean(axis=0)
    gt_std = gt.std(axis=0)
    pred_mean = pred.mean(axis=0)
    pred_std = pred.std(axis=0)
    return (gt - gt_mean) / (gt_std + 1e-4), (pred - pred_mean) / (pred_std + 1e-4)


def calculate_avg_distance(feat):
    feat = np.array(feat)
    n, c = feat.shape
    diff = feat[:, np.newaxis, :] - feat[np.newaxis, :, :]
    sq_diff = np.sum(diff**2, axis=2)
    distances = np.sqrt(sq_diff)
    total_distance = np.sum(np.triu(distances, 1))
    avg_distance = total_distance / ((n * (n - 1)) / 2)
    return avg_distance


def calc_fid(kps_gen, kps_gt):
    kps_gt, kps_gen = np.array(kps_gt), np.array(kps_gen)

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    diff = mu_gen - mu_gt
    eps = 1e-5
    covmean, _ = linalg.sqrtm(sigma_gen.dot(sigma_gt), disp=False)
    if not np.isfinite(covmean).all():
        print(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma_gen.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_gen + offset).dot(sigma_gt + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma_gen) + np.trace(sigma_gt) - 2 * tr_covmean


def calculate_beat_similarity(music_beat, keypoints):
    music_beat, keypoints = np.array(music_beat), np.array(keypoints)
    b, t, _, _ = keypoints.shape
    ba_score = []
    for i in range(b):
        mb = get_mb(music_beat[i])
        db = get_db(keypoints[i])
        ba = BA(mb, db)
        ba_score.append(ba)
    return np.mean(ba_score)


def BA(music_beats, motion_beats):
    if len(music_beats) == 0:
        return 0
    if len(motion_beats[0]) == 0:
        return 0
    ba = 0
    for bb in music_beats:
        ba += np.exp(-np.min((motion_beats[0] - bb) ** 2) / 2 / 9)
    return ba / len(music_beats)


def get_mb(music_beats):
    beats = music_beats.astype(bool)
    beat_axis = np.arange(music_beats.shape[0])
    return beat_axis[beats]


def get_db(keypoints):
    keypoints = keypoints[:, :, :2]
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 3.5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats


def load_valid_json(json_file: Path, key_name: str, length: int):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if key_name not in data:
            print(f"🟡 跳过 {json_file.name}: 不包含键 '{key_name}'")
            return None

        keypoints_array = np.array(data[key_name])

        if keypoints_array.ndim != 3 or keypoints_array.shape[1:] != (25, 3):
            print(f"🟡 跳过 {json_file.name}: keypoints 维度异常 {keypoints_array.shape}")
            return None

        if keypoints_array.shape[0] < length:
            print(f"🟡 跳过 {json_file.name}: 帧数不足 {keypoints_array.shape[0]} < {length}")
            return None

        return keypoints_array[:length, :, :]

    except json.JSONDecodeError:
        print(f"❌ 跳过 {json_file.name}: JSON 格式错误")
        return None
    except Exception as e:
        print(f"❌ 跳过 {json_file.name}: {e}")
        return None


def load_valid_beats(mp4_file: Path, target_sr: int, fixed_length: int, fps: int):
    try:
        y, sr = librosa.load(str(mp4_file), sr=target_sr)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_video_indices = (beat_times * fps).astype(int)

        beat_vector = np.zeros(fixed_length)
        valid_indices = beat_video_indices[beat_video_indices < fixed_length]
        beat_vector[valid_indices] = 1
        return beat_vector
    except Exception as e:
        print(f"❌ 跳过 {mp4_file.name}: beat 提取失败: {e}")
        return None


def collect_aligned_data(gt_path, pred_path, audio_path, json_key, fixed_length, audio_sr, fps):
    gt_dir = Path(gt_path)
    pred_dir = Path(pred_path)
    audio_dir = Path(audio_path)

    if not gt_dir.is_dir():
        raise ValueError(f"GT 路径不存在: {gt_path}")
    if not pred_dir.is_dir():
        raise ValueError(f"Pred 路径不存在: {pred_path}")
    if not audio_dir.is_dir():
        raise ValueError(f"Audio 路径不存在: {audio_path}")

    gt_map = {p.stem: p for p in gt_dir.glob("*.json")}
    pred_map = {p.stem: p for p in pred_dir.glob("*.json")}
    audio_map = {p.stem: p for p in audio_dir.glob("*.mp4")}

    common_stems = sorted(set(gt_map) & set(pred_map) & set(audio_map))

    print(f"GT json 数量   : {len(gt_map)}")
    print(f"Pred json 数量 : {len(pred_map)}")
    print(f"Audio mp4 数量 : {len(audio_map)}")
    print(f"共同样本数量   : {len(common_stems)}")

    if len(common_stems) == 0:
        return None, None, None

    gt_list = []
    pred_list = []
    beat_list = []

    for stem in tqdm(common_stems, desc="对齐并加载样本"):
        gt_arr = load_valid_json(gt_map[stem], json_key, fixed_length)
        pred_arr = load_valid_json(pred_map[stem], json_key, fixed_length)
        beat_arr = load_valid_beats(audio_map[stem], audio_sr, fixed_length, fps)

        if gt_arr is None or pred_arr is None or beat_arr is None:
            continue

        gt_list.append(gt_arr)
        pred_list.append(pred_arr)
        beat_list.append(beat_arr)

    if len(gt_list) == 0:
        return None, None, None

    gt = np.stack(gt_list, axis=0)
    pred = np.stack(pred_list, axis=0)
    beats = np.stack(beat_list, axis=0)
    return gt, pred, beats


def main():
    args = parse_args()

    print("--- JSON 到 NumPy 批量处理与评估脚本 ---")

    gt, pred, beats = collect_aligned_data(
        args.gt_path,
        args.pred_path,
        args.audio_path,
        args.json_key,
        args.fixed_length,
        args.audio_sr,
        args.fps
    )

    if gt is None or pred is None or beats is None:
        print("❌ 数据加载失败，程序终止。")
        return

    print("GT shape:", gt.shape)
    print("Pred shape:", pred.shape)
    print("Beats shape:", beats.shape)

    gt_k = kinetic_features(gt)
    gt_g = geometric_features(gt)

    pred_k = kinetic_features(pred)
    pred_g = geometric_features(pred)

    normalized_gt_k, normalized_pred_k = normalize(gt_k, pred_k)
    normalized_gt_g, normalized_pred_g = normalize(gt_g, pred_g)

    print("\nGeometric Feature:")
    print("Dance Diversity of GT:", calculate_avg_distance(normalized_gt_g))
    print("Dance Diversity of Pred:", calculate_avg_distance(normalized_pred_g))
    print("Dance FID of Pred and GT:", calc_fid(normalized_pred_g, normalized_gt_g))

    print("\nKinetic Feature:")
    print("Dance Diversity of GT:", calculate_avg_distance(normalized_gt_k))
    print("Dance Diversity of Pred:", calculate_avg_distance(normalized_pred_k))
    print("Dance FID of Pred and GT:", calc_fid(normalized_pred_k, normalized_gt_k))

    print("\nBeat Alignment Similarity:")
    print("Beat Similarity of GT:", calculate_beat_similarity(beats, gt))
    print("Beat Similarity of Pred:", calculate_beat_similarity(beats, pred))

    print("\nFeature shapes:")
    print(gt_k.shape, gt_g.shape, pred_k.shape, pred_g.shape)


if __name__ == "__main__":
    main()
