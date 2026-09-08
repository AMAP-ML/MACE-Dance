#!/usr/bin/env python3
import argparse
import colorsys
import json
import math
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SMPL_JOINTS = {
    "root": 0,
    "lhip": 1,
    "rhip": 2,
    "belly": 3,
    "lknee": 4,
    "rknee": 5,
    "spine": 6,
    "lankle": 7,
    "rankle": 8,
    "chest": 9,
    "ltoes": 10,
    "rtoes": 11,
    "neck": 12,
    "linshoulder": 13,
    "rinshoulder": 14,
    "head": 15,
    "lshoulder": 16,
    "rshoulder": 17,
    "lelbow": 18,
    "relbow": 19,
    "lwrist": 20,
    "rwrist": 21,
    "lhand": 22,
    "rhand": 23,
}

OPENPOSE_EDGES = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

OPENPOSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

HAND_COLORS = [
    tuple(int(round(channel * 255)) for channel in colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0))
    for i in range(len(HAND_EDGES))
]
HAND_JOINT_COLOR = (0, 0, 255)
FOOT_EDGES = [(0, 1), (2, 3)]
FOOT_COLORS = [(255, 0, 85), (185, 255, 0)]
FOOT_POINT_COLORS = [FOOT_COLORS[0], FOOT_COLORS[0], FOOT_COLORS[1], FOOT_COLORS[1]]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class SimilarityTransform:
    matrix: np.ndarray
    translate: np.ndarray
    source: str

    def apply(self, points):
        points = np.asarray(points, dtype=np.float32)
        return points @ self.matrix.T + self.translate


@dataclass
class HandTemplate:
    local_points: np.ndarray
    source_scores: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone converter from Motion Expert motion_expert.pkl "
            "(SMPL full_pose) to Wan-Animate/OpenPose-style pose_video.mp4."
        )
    )
    parser.add_argument("--case-dir", type=Path, default=None, help="Optional case directory.")
    parser.add_argument("--motion", type=Path, default=None, help="Path to motion_expert.pkl.")
    parser.add_argument("--motion-name", default="motion_expert.pkl")
    parser.add_argument("--output", type=Path, default=None, help="Path to output pose video.")
    parser.add_argument("--output-name", default="pose_video.mp4")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional output summary json path.")

    parser.add_argument("--reference", type=Path, default=None, help="Reference image/video for alignment.")
    parser.add_argument("--reference-name", default="video.mp4")
    parser.add_argument(
        "--reference-pose-json",
        type=Path,
        default=None,
        help="Optional path to cached reference whole-body keypoints json.",
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument("--reference-mode", choices=("auto", "off", "required"), default="auto")

    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory that contains det/ and pose2d/ ONNX checkpoints for auto reference alignment.",
    )
    parser.add_argument("--det-model", default="det/yolov10m.onnx")
    parser.add_argument("--pose2d-model", default="pose2d/vitpose_h_wholebody.onnx/end2end.onnx")
    parser.add_argument("--det-score-threshold", type=float, default=0.25)
    parser.add_argument("--keypoint-threshold", type=float, default=0.08)
    parser.add_argument("--bbox-scale", type=float, default=1.25)
    parser.add_argument("--reference-align-min-joints", type=int, default=6)
    parser.add_argument(
        "--reference-fit-strategy",
        choices=("first-frame", "canvas-safe"),
        default="canvas-safe",
    )

    parser.add_argument("--num-frames", type=int, default=80, help="<=0 means use all motion frames.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--joint-radius", type=int, default=4)
    parser.add_argument("--hand-line-width", type=int, default=2)
    parser.add_argument("--hand-joint-radius", type=int, default=4)
    parser.add_argument("--hand-scale-ratio", type=float, default=0.55)
    parser.add_argument("--head-scale", type=float, default=1.8)
    parser.add_argument("--head-width-scale", type=float, default=1.65)
    parser.add_argument("--head-height-scale", type=float, default=1.15)
    parser.add_argument("--fit-width-ratio", type=float, default=0.72)
    parser.add_argument("--fit-height-ratio", type=float, default=0.86)
    parser.add_argument("--center-y-ratio", type=float, default=0.56)
    parser.add_argument("--antialias-scale", type=int, default=1)
    parser.add_argument("--body-limb-alpha", type=float, default=0.60)

    parser.set_defaults(draw_hands=True, draw_feet=True, use_reference_hands=True)
    parser.add_argument("--draw-hands", dest="draw_hands", action="store_true")
    parser.add_argument("--no-draw-hands", dest="draw_hands", action="store_false")
    parser.add_argument("--draw-feet", dest="draw_feet", action="store_true")
    parser.add_argument("--no-draw-feet", dest="draw_feet", action="store_false")
    parser.add_argument("--use-reference-hands", dest="use_reference_hands", action="store_true")
    parser.add_argument("--no-use-reference-hands", dest="use_reference_hands", action="store_false")
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        if path is not None and path.is_file():
            return path
    return None


def infer_case_dir(args):
    if args.case_dir is not None:
        return args.case_dir.resolve()
    if args.motion is not None:
        return args.motion.resolve().parent
    if args.output is not None:
        return args.output.resolve().parent
    return None


def resolve_motion_path(args, case_dir):
    if args.motion is not None:
        return args.motion.resolve()
    if case_dir is not None:
        return (case_dir / args.motion_name).resolve()
    raise ValueError("Please provide --motion or --case-dir")


def resolve_output_path(args, case_dir, motion_path):
    if args.output is not None:
        return args.output.resolve()
    if case_dir is not None:
        return (case_dir / args.output_name).resolve()
    return motion_path.with_name(args.output_name).resolve()


def resolve_summary_path(args, output_path):
    if args.summary_json is not None:
        return args.summary_json.resolve()
    return output_path.with_name("pose_video_summary.json")


def resolve_reference_path(args, case_dir):
    if args.reference is not None:
        return args.reference.resolve()
    if case_dir is None:
        return None
    candidates = [
        case_dir / args.reference_name,
        case_dir / "video.mp4",
        case_dir / "first_frame.jpg",
        case_dir / "first_frame.jpeg",
        case_dir / "first_frame.png",
        case_dir / "reference.jpg",
        case_dir / "reference.jpeg",
        case_dir / "reference.png",
        case_dir / "ref.jpg",
        case_dir / "ref.jpeg",
        case_dir / "ref.png",
        case_dir / "rgb.mp4",
    ]
    return first_existing(candidates)


def resolve_reference_pose_json_path(args, case_dir, output_path):
    if args.reference_pose_json is not None:
        return args.reference_pose_json.resolve()
    if case_dir is not None:
        return (case_dir / "reference_pose_wholebody.json").resolve()
    return output_path.with_name("reference_pose_wholebody.json")


def load_full_pose(motion_path):
    with motion_path.open("rb") as f:
        motion = pickle.load(f)
    if not isinstance(motion, dict) or "full_pose" not in motion:
        raise KeyError(f"{motion_path} does not contain key 'full_pose'")

    poses = np.asarray(motion["full_pose"], dtype=np.float32)
    while poses.ndim > 3 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or poses.shape[1:] != (24, 3):
        raise ValueError(f"Expected full_pose shape (T, 24, 3), got {poses.shape}")
    return poses


def select_frames(poses, num_frames):
    if num_frames > 0:
        return poses[:num_frames]
    return poses


def normalize(vec, fallback=(1.0, 0.0)):
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def raw_smpl_2d(pose):
    coords = pose[:, [0, 2]].astype(np.float32)
    coords[:, 1] = -coords[:, 1]
    return coords


def smpl_to_openpose(points, head_scale=1.0, head_width_scale=1.0, head_height_scale=1.0):
    j = SMPL_JOINTS
    op = np.ones((18, 3), dtype=np.float32)
    op[:, :2] = 0.0

    lshoulder = points[j["lshoulder"]]
    rshoulder = points[j["rshoulder"]]
    neck = (lshoulder + rshoulder) * 0.5
    head = points[j["head"]]
    head_dir = normalize(head - neck, fallback=(0.0, -1.0))
    shoulder_dir = normalize(lshoulder - rshoulder)
    shoulder_width = max(float(np.linalg.norm(lshoulder - rshoulder)), 1e-4)

    head_scale = max(float(head_scale), 0.05)
    head_width = head_scale * max(float(head_width_scale), 0.05)
    head_height = head_scale * max(float(head_height_scale), 0.05)
    nose = head + head_dir * shoulder_width * 0.10 * head_height
    left_eye = nose + shoulder_dir * shoulder_width * 0.055 * head_width + head_dir * shoulder_width * 0.018 * head_height
    right_eye = nose - shoulder_dir * shoulder_width * 0.055 * head_width + head_dir * shoulder_width * 0.018 * head_height
    left_ear = nose + shoulder_dir * shoulder_width * 0.095 * head_width
    right_ear = nose - shoulder_dir * shoulder_width * 0.095 * head_width

    mapping = {
        0: nose,
        1: neck,
        2: points[j["rshoulder"]],
        3: points[j["relbow"]],
        4: points[j["rwrist"]],
        5: points[j["lshoulder"]],
        6: points[j["lelbow"]],
        7: points[j["lwrist"]],
        8: points[j["rhip"]],
        9: points[j["rknee"]],
        10: points[j["rankle"]],
        11: points[j["lhip"]],
        12: points[j["lknee"]],
        13: points[j["lankle"]],
        14: right_eye,
        15: left_eye,
        16: right_ear,
        17: left_ear,
    }
    for idx, value in mapping.items():
        op[idx, :2] = value
    return op


def vitpose_to_openpose(keypoints):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    op = np.zeros((18, 3), dtype=np.float32)
    coco_to_openpose = {
        0: 0,
        2: 6,
        3: 8,
        4: 10,
        5: 5,
        6: 7,
        7: 9,
        8: 12,
        9: 14,
        10: 16,
        11: 11,
        12: 13,
        13: 15,
        14: 2,
        15: 1,
        16: 4,
        17: 3,
    }
    for op_idx, coco_idx in coco_to_openpose.items():
        op[op_idx] = keypoints[coco_idx]

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    if left_shoulder[2] > 0 and right_shoulder[2] > 0:
        op[1, :2] = (left_shoulder[:2] + right_shoulder[:2]) * 0.5
        op[1, 2] = min(left_shoulder[2], right_shoulder[2])
    return op


def extract_keypoints_array(reference_pose):
    if reference_pose is None:
        return None
    if isinstance(reference_pose, dict):
        if "keypoints" in reference_pose:
            return np.asarray(reference_pose["keypoints"], dtype=np.float32)
        if "openpose_keypoints" in reference_pose:
            return np.asarray(reference_pose["openpose_keypoints"], dtype=np.float32)
    if isinstance(reference_pose, list):
        return np.asarray(reference_pose, dtype=np.float32)
    raise ValueError("Reference pose json must contain 'keypoints' or be a raw keypoint list")


def reference_to_openpose(reference_pose):
    keypoints = extract_keypoints_array(reference_pose)
    if keypoints is None:
        raise ValueError("Reference pose is empty")
    if keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"Expected reference keypoints shape (N, 3), got {keypoints.shape}")
    if keypoints.shape[0] == 18:
        return keypoints.astype(np.float32)
    if keypoints.shape[0] >= 133:
        return vitpose_to_openpose(keypoints[:133])
    raise ValueError(f"Unsupported reference keypoint count: {keypoints.shape[0]}")


def estimate_similarity(src, dst):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_var = np.sum(src_centered ** 2)
    if src_var < 1e-8:
        raise ValueError("Source points are degenerate")
    covariance = src_centered.T @ dst_centered
    u, singular, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = float(np.sum(singular) / src_var)
    translate = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale * rotation, translate.astype(np.float32), "reference-pose")


def build_canvas_fit_transform(poses, width, height, fit_width_ratio, fit_height_ratio, center_y_ratio):
    coords = np.stack([raw_smpl_2d(pose) for pose in poses], axis=0)
    mins = coords.reshape(-1, 2).min(axis=0)
    maxs = coords.reshape(-1, 2).max(axis=0)
    center = (mins + maxs) * 0.5
    span = np.maximum(maxs - mins, 1e-6)
    scale = min(width * fit_width_ratio / span[0], height * fit_height_ratio / span[1])
    canvas_center = np.array([width * 0.5, height * center_y_ratio], dtype=np.float32)
    matrix = np.eye(2, dtype=np.float32) * float(scale)
    translate = canvas_center - matrix @ center
    return SimilarityTransform(matrix, translate, "canvas-fit")


def keep_sequence_in_canvas(transform, poses, width, height, fit_width_ratio, fit_height_ratio, center_y_ratio):
    raw = np.stack([raw_smpl_2d(pose) for pose in poses], axis=0).reshape(-1, 2)
    points = transform.apply(raw)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale_down = min(width * fit_width_ratio / span[0], height * fit_height_ratio / span[1], 1.0)
    margin = max(width, height) * 0.02
    out_of_bounds = mins[0] < margin or mins[1] < margin or maxs[0] > width - margin or maxs[1] > height - margin
    if scale_down >= 0.999 and not out_of_bounds:
        return transform

    old_center = (mins + maxs) * 0.5
    desired_center = np.array([width * 0.5, height * center_y_ratio], dtype=np.float32)
    matrix = transform.matrix * float(scale_down)
    translate = desired_center + float(scale_down) * (transform.translate - old_center)
    return SimilarityTransform(matrix, translate.astype(np.float32), transform.source + "+canvas-safe")


def build_reference_transform(poses, reference_pose, args):
    source_openpose = smpl_to_openpose(raw_smpl_2d(poses[0]))
    reference_openpose = reference_to_openpose(reference_pose)
    stable_indices = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
    valid = reference_openpose[stable_indices, 2] >= args.keypoint_threshold
    if int(valid.sum()) < args.reference_align_min_joints:
        raise ValueError(f"Only {int(valid.sum())} valid reference body joints")

    transform = estimate_similarity(
        source_openpose[stable_indices[valid], :2],
        reference_openpose[stable_indices[valid], :2],
    )
    if args.reference_fit_strategy == "canvas-safe":
        transform = keep_sequence_in_canvas(
            transform,
            poses,
            args.width,
            args.height,
            args.fit_width_ratio,
            args.fit_height_ratio,
            args.center_y_ratio,
        )
    return transform


def build_hand_templates(reference_pose, args):
    if not reference_pose or not args.use_reference_hands:
        return {}
    keypoints = extract_keypoints_array(reference_pose)
    if keypoints is None or keypoints.shape[0] < 133:
        return {}

    templates = {}
    for side, start in (("left", 91), ("right", 112)):
        hand = keypoints[start : start + 21]
        if hand.shape != (21, 3):
            continue
        valid = hand[:, 2] >= args.keypoint_threshold
        if int(valid.sum()) < 6 or hand[0, 2] < args.keypoint_threshold:
            continue
        wrist = hand[0, :2]
        tip_indices = np.array([4, 8, 12, 16, 20])
        valid_tips = tip_indices[hand[tip_indices, 2] >= args.keypoint_threshold]
        if len(valid_tips) > 0:
            axis = hand[valid_tips, :2].mean(axis=0) - wrist
        else:
            axis = hand[9, :2] - wrist
        axis_len = float(np.linalg.norm(axis))
        if axis_len < 1e-4:
            coords = hand[valid, :2]
            span = coords.max(axis=0) - coords.min(axis=0)
            axis_len = max(float(np.linalg.norm(span)) * 0.5, 1.0)
            axis = np.array([axis_len, 0.0], dtype=np.float32)
        axis_unit = normalize(axis)
        perp_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)
        offsets = hand[:, :2] - wrist
        local_xy = np.stack([offsets @ axis_unit, offsets @ perp_unit], axis=1) / axis_len
        local_points = np.ones((21, 3), dtype=np.float32)
        local_points[:, :2] = local_xy
        local_points[:, 2] = hand[:, 2]
        templates[side] = HandTemplate(local_points=local_points, source_scores=hand[:, 2].copy())
    return templates


def make_hand(points, side, args, template=None):
    j = SMPL_JOINTS
    if side == "left":
        wrist = points[j["lwrist"]]
        hand = points[j["lhand"]]
        elbow = points[j["lelbow"]]
        sign = 1.0
    else:
        wrist = points[j["rwrist"]]
        hand = points[j["rhand"]]
        elbow = points[j["relbow"]]
        sign = -1.0

    forearm = max(float(np.linalg.norm(wrist - elbow)), 1.0)
    forward = hand - wrist
    if np.linalg.norm(forward) < forearm * 0.08:
        forward = wrist - elbow
    forward = normalize(forward)
    side_vec = np.array([-forward[1], forward[0]], dtype=np.float32) * sign

    if template is not None:
        current_perp = np.array([-forward[1], forward[0]], dtype=np.float32)
        current_scale = forearm * args.hand_scale_ratio
        hand_points = np.ones((21, 3), dtype=np.float32)
        hand_points[:, :2] = (
            wrist
            + template.local_points[:, :1] * current_scale * forward
            + template.local_points[:, 1:2] * current_scale * current_perp
        )
        hand_points[:, 2] = template.source_scores
        return hand_points

    base_len = forearm * args.hand_scale_ratio
    angles = [-30, -14, 0, 14, 30]
    lengths = [0.70, 0.88, 0.96, 0.86, 0.70]
    hand_points = np.ones((21, 3), dtype=np.float32)
    hand_points[0, :2] = wrist
    cursor = 1
    for angle, length_scale in zip(angles, lengths):
        rad = math.radians(angle)
        direction = normalize(math.cos(rad) * forward + math.sin(rad) * side_vec)
        segment = base_len * length_scale / 4.0
        start = wrist + side_vec * base_len * 0.10 * (angle / 42.0)
        for joint_idx in range(4):
            hand_points[cursor, :2] = start + direction * segment * (joint_idx + 1)
            cursor += 1
    return hand_points


def is_visible(point, threshold):
    return np.isfinite(point[:2]).all() and point[2] >= threshold


def cv_point(point):
    return tuple(np.round(point[:2]).astype(np.int32).tolist())


def draw_capsule_links(canvas, points, edges, colors, stick_width, threshold):
    stick_width = max(1, int(stick_width))
    points = np.asarray(points, dtype=np.float32)
    for edge_idx, edge in enumerate(edges):
        a, b = edge
        if not is_visible(points[a], threshold) or not is_visible(points[b], threshold):
            continue
        x = points[[a, b], 0]
        y = points[[a, b], 1]
        length = float(np.linalg.norm(points[a, :2] - points[b, :2]))
        if length < 1e-3:
            continue
        center = (int(np.mean(x)), int(np.mean(y)))
        angle = math.degrees(math.atan2(float(y[0] - y[1]), float(x[0] - x[1])))
        polygon = cv2.ellipse2Poly(center, (int(length / 2), stick_width), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[edge_idx % len(colors)])


def draw_keypoints(canvas, points, colors, radius, threshold):
    radius = max(0, int(radius))
    if radius <= 0:
        return
    points = np.asarray(points, dtype=np.float32)
    for point_idx, point in enumerate(points):
        if not is_visible(point, threshold):
            continue
        cv2.circle(canvas, cv_point(point), radius, colors[point_idx % len(colors)], thickness=-1)


def draw_handpose(canvas, hands, line_width, radius, threshold):
    line_width = max(1, int(line_width))
    radius = max(0, int(radius))
    for hand in hands:
        hand = np.asarray(hand, dtype=np.float32)
        for edge_idx, edge in enumerate(HAND_EDGES):
            a, b = edge
            if not is_visible(hand[a], threshold) or not is_visible(hand[b], threshold):
                continue
            x1, y1 = cv_point(hand[a])
            x2, y2 = cv_point(hand[b])
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(canvas, (x1, y1), (x2, y2), HAND_COLORS[edge_idx], thickness=line_width)
        for point in hand:
            if not is_visible(point, threshold):
                continue
            x, y = cv_point(point)
            if x > 0 and y > 0 and radius > 0:
                cv2.circle(canvas, (x, y), radius, HAND_JOINT_COLOR, thickness=-1)


def draw_frame(pose, transform, args, hand_templates=None):
    aa = max(1, int(args.antialias_scale))
    canvas = np.zeros((args.height * aa, args.width * aa, 3), dtype=np.uint8)
    smpl_points = transform.apply(raw_smpl_2d(pose))
    smpl_points_aa = smpl_points * aa
    openpose = smpl_to_openpose(
        smpl_points_aa,
        head_scale=args.head_scale,
        head_width_scale=args.head_width_scale,
        head_height_scale=args.head_height_scale,
    )

    body_width = max(1, args.line_width * aa)
    body_radius = max(0, args.joint_radius * aa)
    limb_canvas = canvas.copy()
    draw_capsule_links(limb_canvas, openpose, OPENPOSE_EDGES, OPENPOSE_COLORS, body_width, 0.0)

    if args.draw_feet:
        j = SMPL_JOINTS
        foot_points = np.ones((4, 3), dtype=np.float32)
        foot_points[:, :2] = [
            smpl_points_aa[j["lankle"]],
            smpl_points_aa[j["ltoes"]],
            smpl_points_aa[j["rankle"]],
            smpl_points_aa[j["rtoes"]],
        ]
        draw_capsule_links(limb_canvas, foot_points, FOOT_EDGES, FOOT_COLORS, body_width, 0.0)

    body_alpha = float(np.clip(args.body_limb_alpha, 0.0, 1.0))
    canvas = (limb_canvas.astype(np.float32) * body_alpha).astype(np.uint8)
    draw_keypoints(canvas, openpose, OPENPOSE_COLORS, body_radius, 0.0)
    if args.draw_feet:
        draw_keypoints(canvas, foot_points, FOOT_POINT_COLORS, body_radius, 0.0)

    if args.draw_hands:
        hand_templates = hand_templates or {}
        hands = []
        thresholds = []
        for side in ("left", "right"):
            hand = make_hand(smpl_points_aa, side, args, template=hand_templates.get(side))
            hands.append(hand)
            thresholds.append(args.keypoint_threshold if hand_templates.get(side) is not None else 0.0)
        for hand, threshold in zip(hands, thresholds):
            draw_handpose(
                canvas,
                [hand],
                max(1, args.hand_line_width * aa),
                max(0, args.hand_joint_radius * aa),
                threshold,
            )

    if aa > 1:
        canvas = cv2.resize(canvas, (args.width, args.height), interpolation=cv2.INTER_AREA)
    return canvas


def write_video_with_ffmpeg(frames, output_path, fps, quality):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    if not frames:
        raise ValueError("No frames to write")

    crf = int(np.clip(30 - quality, 16, 30))
    with tempfile.TemporaryDirectory(prefix="pose_video_frames_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for frame_idx, frame in enumerate(frames):
            frame_path = tmp_dir / f"frame_{frame_idx:06d}.png"
            if not cv2.imwrite(str(frame_path), frame[:, :, ::-1]):
                raise RuntimeError(f"Could not write temporary frame {frame_path}")

        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(tmp_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            str(output_path),
        ]
        subprocess.run(command, check=True)


def write_video_with_cv2(frames, output_path, fps):
    if not frames:
        raise ValueError("No frames to write")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(frame[:, :, ::-1])
    finally:
        writer.release()


def write_video(frames, output_path, fps, quality):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except Exception:
        try:
            write_video_with_ffmpeg(frames, output_path, fps, quality)
        except Exception:
            write_video_with_cv2(frames, output_path, fps)
        return

    writer = imageio.get_writer(str(output_path), fps=fps, quality=quality, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def load_reference_pose_json(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"reference": str(path), "keypoints": data}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported reference json format: {path}")


class ReferencePoseExtractor:
    def __init__(self, args):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise RuntimeError("Automatic reference pose extraction requires onnxruntime") from exc

        if args.ckpt_dir is None:
            raise ValueError("--ckpt-dir is required for automatic reference alignment")

        det_path = args.ckpt_dir.resolve() / args.det_model
        pose_path = args.ckpt_dir.resolve() / args.pose2d_model
        if not det_path.is_file():
            raise FileNotFoundError(det_path)
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)

        providers = ["CPUExecutionProvider"]
        self.det_session = ort.InferenceSession(str(det_path), providers=providers)
        self.pose_session = ort.InferenceSession(str(pose_path), providers=providers)
        self.det_input = self.det_session.get_inputs()[0].name
        self.pose_input = self.pose_session.get_inputs()[0].name
        self.args = args

    def read_image(self, path, frame_index):
        if path.suffix.lower() in VIDEO_SUFFIXES:
            cap = cv2.VideoCapture(str(path))
            if frame_index > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise ValueError(f"Could not read frame {frame_index} from {path}")
            return frame
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image {path}")
        return image

    def extract_from_path(self, reference_path):
        image = self.read_image(reference_path, self.args.reference_frame)
        bbox, det_score = self.detect_person(image)
        keypoints = self.estimate_pose(image, bbox)
        return {
            "reference": str(reference_path),
            "image_size": [int(image.shape[1]), int(image.shape[0])],
            "bbox_xyxy": [float(x) for x in bbox],
            "det_score": float(det_score),
            "keypoints": keypoints.astype(float).tolist(),
        }

    def letterbox(self, image, size=640):
        h, w = image.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = (size - new_w) // 2
        pad_h = (size - new_h) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
        return canvas, scale, pad_w, pad_h

    def detect_person(self, image):
        canvas, scale, pad_w, pad_h = self.letterbox(image)
        inp = canvas[:, :, ::-1].astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]
        pred = self.det_session.run(None, {self.det_input: inp})[0][0]
        pred = pred[np.isfinite(pred).all(axis=1)]
        pred = pred[(pred[:, 4] >= self.args.det_score_threshold) & (pred[:, 5].astype(np.int64) == 0)]
        if len(pred) == 0:
            raise ValueError("No person detected in reference image")
        areas = np.maximum(pred[:, 2] - pred[:, 0], 0) * np.maximum(pred[:, 3] - pred[:, 1], 0)
        best = pred[np.argmax(pred[:, 4] * np.sqrt(np.maximum(areas, 1.0)))]
        x1, y1, x2, y2, score, _ = best
        bbox = np.array(
            [(x1 - pad_w) / scale, (y1 - pad_h) / scale, (x2 - pad_w) / scale, (y2 - pad_h) / scale],
            dtype=np.float32,
        )
        h, w = image.shape[:2]
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h - 1)
        return bbox, score

    def estimate_pose(self, image, bbox):
        x1, y1, x2, y2 = bbox.astype(np.float32)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        box_w, box_h = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
        aspect = 192 / 256
        if box_w / box_h > aspect:
            box_h = box_w / aspect
        else:
            box_w = box_h * aspect
        box_w *= self.args.bbox_scale
        box_h *= self.args.bbox_scale
        x1, x2 = cx - box_w * 0.5, cx + box_w * 0.5
        y1, y2 = cy - box_h * 0.5, cy + box_h * 0.5
        crop = self.crop_with_padding(image, x1, y1, x2, y2)
        resized = cv2.resize(crop, (192, 256), interpolation=cv2.INTER_LINEAR)
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = ((rgb - mean) / std).transpose(2, 0, 1)[None]
        heatmaps = self.pose_session.run(None, {self.pose_input: inp})[0][0]
        return self.decode_heatmaps(heatmaps, x1, y1, x2, y2)

    def crop_with_padding(self, image, x1, y1, x2, y2):
        ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
        ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
        crop = np.zeros((max(iy2 - iy1, 1), max(ix2 - ix1, 1), 3), dtype=np.uint8)
        h, w = image.shape[:2]
        sx1, sy1 = max(0, ix1), max(0, iy1)
        sx2, sy2 = min(w, ix2), min(h, iy2)
        if sx2 <= sx1 or sy2 <= sy1:
            return crop
        dx1, dy1 = sx1 - ix1, sy1 - iy1
        crop[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = image[sy1:sy2, sx1:sx2]
        return crop

    def decode_heatmaps(self, heatmaps, x1, y1, x2, y2):
        keypoints = np.zeros((heatmaps.shape[0], 3), dtype=np.float32)
        heat_h, heat_w = heatmaps.shape[1:]
        for idx, heatmap in enumerate(heatmaps):
            flat_idx = int(np.argmax(heatmap))
            py, px = divmod(flat_idx, heat_w)
            score = float(heatmap[py, px])
            px = float(px)
            py = float(py)
            if 1 <= px < heat_w - 1 and 1 <= py < heat_h - 1:
                dx = heatmap[int(py), int(px) + 1] - heatmap[int(py), int(px) - 1]
                dy = heatmap[int(py) + 1, int(px)] - heatmap[int(py) - 1, int(px)]
                px += float(np.sign(dx)) * 0.25
                py += float(np.sign(dy)) * 0.25
            crop_x = (px + 0.5) * 192.0 / heat_w
            crop_y = (py + 0.5) * 256.0 / heat_h
            keypoints[idx, 0] = x1 + crop_x / 192.0 * (x2 - x1)
            keypoints[idx, 1] = y1 + crop_y / 256.0 * (y2 - y1)
            keypoints[idx, 2] = score
        return keypoints


def load_or_extract_reference_pose(args, reference_path, reference_pose_json):
    if reference_pose_json is not None and reference_pose_json.is_file():
        return load_reference_pose_json(reference_pose_json)

    if reference_path is None:
        raise FileNotFoundError("Reference image/video was not provided or found")

    extractor = ReferencePoseExtractor(args)
    result = extractor.extract_from_path(reference_path)
    if reference_pose_json is not None:
        reference_pose_json.parent.mkdir(parents=True, exist_ok=True)
        with reference_pose_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return result


def render_pose_video(args):
    case_dir = infer_case_dir(args)
    motion_path = resolve_motion_path(args, case_dir)
    output_path = resolve_output_path(args, case_dir, motion_path)
    summary_path = resolve_summary_path(args, output_path)
    reference_path = resolve_reference_path(args, case_dir)
    reference_pose_json = resolve_reference_pose_json_path(args, case_dir, output_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)

    poses_all = load_full_pose(motion_path)
    poses = select_frames(poses_all, args.num_frames)
    if len(poses) == 0:
        raise ValueError(f"No frames selected from {motion_path}")

    transform = build_canvas_fit_transform(
        poses,
        args.width,
        args.height,
        args.fit_width_ratio,
        args.fit_height_ratio,
        args.center_y_ratio,
    )

    reference_pose = None
    reference_error = None
    if args.reference_mode != "off":
        try:
            reference_pose = load_or_extract_reference_pose(args, reference_path, reference_pose_json)
            transform = build_reference_transform(poses, reference_pose, args)
        except Exception as exc:
            reference_error = str(exc)
            if args.reference_mode == "required":
                raise
            print(f"[reference disabled] {exc}")

    hand_templates = build_hand_templates(reference_pose, args)
    frames = [draw_frame(pose, transform, args, hand_templates=hand_templates) for pose in poses]
    write_video(frames, output_path, args.fps, args.quality)

    summary = {
        "status": "ok",
        "motion": str(motion_path),
        "output": str(output_path),
        "frames": len(poses),
        "source_frames": len(poses_all),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "transform_source": transform.source,
        "transform_matrix": transform.matrix.astype(float).tolist(),
        "transform_translate": transform.translate.astype(float).tolist(),
        "reference": str(reference_path) if reference_path is not None else None,
        "reference_pose_json": str(reference_pose_json) if reference_pose_json is not None else None,
        "reference_error": reference_error,
        "reference_hand_templates": sorted(hand_templates.keys()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return summary


def main():
    args = parse_args()
    summary = render_pose_video(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()