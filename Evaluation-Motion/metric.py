import os
import numpy as np
import torch
import pickle
from vis import SMPLSkeleton
from einops import rearrange
from model.features.geometric import geometric_features
from model.features.kinetic import kinetic_features
from model.cls import CLS
import json

style2s = [
    'Dai', 'ShenYun', 'Wei', 'Korean', 'Urban', 'Hiphop', 'Popping', 'Miao', 'HanTang', \
    'Breaking', 'Kun', 'Locking', 'Jazz', 'Choreography', 'Chinese', 'DunHuang'
]

from pytorch3d.transforms import RotateAxisAngle, axis_angle_to_quaternion, quaternion_multiply, quaternion_to_axis_angle


class Metric:
    def __init__(self, gt_dir, device):
        self.smpl = SMPLSkeleton(device)
        music_dir = os.path.join(gt_dir, 'baseline_feats')
        motion_dir = os.path.join(gt_dir, 'motions_sliced')
        label_dir = os.path.join(gt_dir, 'label')

        music_feat, smpl_poses, smpl_trans, label = [], [], [], []
        for filename in sorted(os.listdir(music_dir)):
            music_feat.append(np.load(os.path.join(music_dir, filename)))
            with open(os.path.join(motion_dir, filename.replace('.npy', '.pkl')), "rb") as f:
                motion = pickle.load(f)
                smpl_poses.append(motion['q'])
                smpl_trans.append(motion['pos'])
            with open(os.path.join(label_dir, filename[:3]+'.json'), 'r') as f:
                label.append(style2s.index(json.load(f).get('style2')))

        music_feat, smpl_trans, smpl_poses = np.array(music_feat), np.array(smpl_trans), np.array(smpl_poses)
        music_feat, smpl_trans, smpl_poses = torch.from_numpy(music_feat), torch.from_numpy(smpl_trans), torch.from_numpy(smpl_poses)
        smpl_trans, smpl_poses = smpl_trans.to(device), smpl_poses.to(device)
        smpl_poses = rearrange(smpl_poses, 'b t (c1 c2) -> b t c1 c2', c2=3)
        keypoint = self.smpl.forward(smpl_poses, smpl_trans).detach().cpu()
        keypoint = align_axis(keypoint)
        label = torch.from_numpy(np.array(label))

        self.geo_feat_gt = geometric_features(keypoint)
        self.kin_feat_gt = kinetic_features(keypoint)
        self.pfc_gt = calculate_pfc(keypoint)

        self.keypoint_gt = keypoint
        self.music_beat_gt = music_feat[:, :, 34]
        self.label = label


    def calculate_metric(self, pred_dir, device):
        keypoint, smpl_trans, smpl_poses = [], [], []
        for filename in sorted(os.listdir(pred_dir)):
            with open(os.path.join(pred_dir, filename), "rb") as f:
                data = pickle.load(f)
                smpl_trans.append(data['smpl_trans'])
                smpl_poses.append(data['smpl_poses'])
                keypoint.append(data['full_pose'])
        smpl_trans, smpl_poses = torch.from_numpy(np.array(smpl_trans)), torch.from_numpy(np.array(smpl_poses))
        smpl_poses = rearrange(smpl_poses, 'b t (c1 c2) -> b t c1 c2', c2=3)
        smpl_trans, smpl_poses = smpl_trans.to(device), smpl_poses.to(device)
        smpl_poses, smpl_trans = rotate_pred_like_gt(smpl_poses, smpl_trans)
        keypoint = self.smpl.forward(smpl_poses, smpl_trans).detach().cpu()
        keypoint = align_axis(keypoint)


        kin_feat_pred = kinetic_features(keypoint)
        geo_feat_pred = geometric_features(keypoint)


        print('Beat Alignment Similarity:')
        print('Beat Similarity of GT:', calculate_beat_similarity(self.music_beat_gt, self.keypoint_gt))
        print('Beat Similarity of Pred:', calculate_beat_similarity(self.music_beat_gt, keypoint))

        print('Geometric Feature:')
        dance_gt_geometric_feature, dance_pred_geometric_feature = self.normalize(self.geo_feat_gt, geo_feat_pred)
        print('Dance Diversity of GT:', calculate_avg_distance(dance_gt_geometric_feature))
        print('Dance Diversity of Pred:', calculate_avg_distance(dance_pred_geometric_feature))
        print('Dance FID of Pred and GT:', calc_fid(dance_pred_geometric_feature, dance_gt_geometric_feature))

        print('Kinetic Feature:')
        dance_gt_kinetic_feature, dance_pred_kinetic_feature = self.normalize(self.kin_feat_gt, kin_feat_pred)
        # gt_max, _ = torch.max(kin_feat_gt, dim=0)
        # pred_max, _ = torch.max(kin_feat_pred, dim=0)
        # gt_min, _ = torch.min(kin_feat_gt, dim=0)
        # pred_min, _ = torch.min(kin_feat_pred, dim=0)
        # print('gt', gt_max)
        # print('pd', pred_max)
        # print('gt', gt_min)
        # print('pd', pred_min)
        # import sys
        # sys.exit()
        print('Dance Diversity of GT:', calculate_avg_distance(dance_gt_kinetic_feature))
        print('Dance Diversity of Pred:', calculate_avg_distance(dance_pred_kinetic_feature))
        print('Dance FID of Pred and GT:', calc_fid(dance_pred_kinetic_feature, dance_gt_kinetic_feature))

        print('PFC:')
        print('PFC of GT:', self.pfc_gt)
        print('PFC of Pred:', calculate_pfc(keypoint))


    def normalize(self, gt, pred):
        mean = gt.mean(axis=0)
        std = gt.std(axis=0)
        return (gt - mean) / (std + 1e-4), (pred - mean) / (std + 1e-4)


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model

from tqdm import tqdm
from scipy import linalg
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
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
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def get_mb(music_beats):
    t = music_beats.shape[0]
    beats = music_beats.astype(bool)
    beat_axis = np.arange(t)
    beat_axis = beat_axis[beats]
    return beat_axis

def get_db(keypoints):
    t, _, _ = keypoints.shape
    keypoints = keypoints.reshape(t, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats


def calc_fid(kps_gen, kps_gt):

    kps_gt, kps_gen = np.array(kps_gt), np.array(kps_gen)

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_avg_distance(feat):
    feat = np.array(feat)
    n, c = feat.shape
    diff = feat[:, np.newaxis, :] - feat[np.newaxis, :, :]
    sq_diff = np.sum(diff**2, axis=2)
    distances = np.sqrt(sq_diff)
    total_distance = np.sum(np.triu(distances, 1))
    avg_distance = total_distance / ((n * (n - 1)) / 2)
    return avg_distance

def compute_acc(pred_logits, gt_labels):
    pred_classes = torch.argmax(pred_logits, dim=1)
    
    pred_np = pred_classes.cpu().numpy()
    gt_np = gt_labels.cpu().numpy()
    
    correct = (pred_classes == gt_labels).sum().item()
    accuracy = correct / gt_labels.size(0)
    return accuracy

def calculate_pfc(joints):
    scores = []
    up_dir = 2  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30
    for i in range(len(joints)):
        joint3d = joints[i]
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
    out = np.mean(scores) * 10000
    return out

def rotate_pred_like_gt(local_q, root_pos):
    """
    把 Pred 从 Z-up 坐标系转回到 Y-up，和原始 GT 对齐
    local_q: (B, T, J, 3)  axis-angle
    root_pos: (B, T, 3)
    """
    local_q = local_q.float()
    root_pos = root_pos.float()

    # step1: 只取 root joint 的旋转
    root_q = local_q[:, :, :1, :]  # (B, T, 1, 3)
    root_q_quat = axis_angle_to_quaternion(root_q)

    # step2: 乘上绕 X -90° 的旋转
    rotation = torch.tensor(
        [0.7071068, -0.7071068, 0, 0],  # cos(-45)=0.707, sin(-45)=-0.707
        dtype=torch.float32,
        device=root_q_quat.device
    ).unsqueeze(0)  # 保证 batch 对齐
    root_q_quat = quaternion_multiply(rotation, root_q_quat)

    # step3: 转回 axis-angle
    root_q = quaternion_to_axis_angle(root_q_quat)
    local_q[:, :, :1, :] = root_q

    # step4: 旋转 root 平移
    pos_rotation = RotateAxisAngle(-90, axis="X", degrees=True, device=root_pos.device)
    root_pos = pos_rotation.transform_points(root_pos)

    return local_q, root_pos

def align_axis(keypoints: torch.Tensor) -> torch.Tensor:
    """
    keypoints: (B, T, J, 3)
    把 Z-up 的 keypoints 旋转到 -Y
    """
    keypoints = keypoints.float()
    B, T, J, C = keypoints.shape
    rot = RotateAxisAngle(90, axis="z", degrees=True, device=keypoints.device)

    flat = keypoints.reshape(-1, 3)          # (B*T*J, 3)
    rotated = rot.transform_points(flat)     # (B*T*J, 3)
    return rotated.reshape(B, T, J, C)       # (B, T, J, 3)