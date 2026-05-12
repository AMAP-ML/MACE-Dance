# BSD License (original license retained)
# ... (rest of the license) ...
# Modified by Ruilong Li
# Adapted for 2D ViTPose data by an AI assistant

import numpy as np
import torch

# [VITPOSE ADAPTATION]
# Define the new joint set from ViTPose (25 joints)
VITPOSE_25_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',           # 0-4
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',     # 5-8
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',               # 9-12
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',             # 13-16
    'head_top', 'neck_viz',                                            # 17-18
    'left_big_toe', 'left_small_toe', 'left_heel',                      # 19-21
    'right_big_toe', 'right_small_toe', 'right_heel',                   # 22-24
]

# The original SMPL names that the feature functions expect
SMPL_NAMES_NEEDED = [
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", 
    "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", 
    "head",  "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", 
    "lhand", "rhand",
]

# The 2D-adapted utility functions from the previous answer
class feat_utils:
    # ... (Keep the entire 2D feat_utils class from the previous answer here) ...
    @staticmethod
    def distance_between_points(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    @staticmethod
    def angle_within_range(p1, p2, p3, p4, angle_range):
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p4) - np.array(p3)
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
        return 1.0 if angle_range[0] <= angle <= angle_range[1] else 0.0
    @staticmethod
    def velocity_above_threshold(p1, p1_prev, threshold):
        velocity = feat_utils.distance_between_points(p1, p1_prev)
        return 1.0 if velocity > threshold else 0.0
    @staticmethod
    def velocity_direction_above_threshold(p1, p1_prev, j2, j2_prev, j3, j3_prev, range):
        vel1 = np.array(p1) - np.array(p1_prev)
        return 1.0 if np.linalg.norm(vel1) > range else 0.0
    @staticmethod
    def velocity_direction_above_threshold_normal(p1, p1_prev, j2, j3, j4, j4_prev, range):
        vel1 = np.array(p1) - np.array(p1_prev)
        return 1.0 if np.linalg.norm(vel1) > range else 0.0
    @staticmethod
    def distance_from_plane(j1, j2, j3, j4, threshold): return 0.0
    @staticmethod
    def distance_from_plane_normal(j1, j2, j3, j4, threshold): return 0.0


# [VITPOSE ADAPTATION]
# A new class specifically for handling the mapping from ViTPose to SMPL semantics
class GeometricFeaturesViTPose:
    def __init__(self, vitpose_positions_2d):
        # vitpose_positions_2d: (t, 25, 2) numpy array
        self.frame_num = 1
        
        # --- Create a new, extended set of joints that matches SMPL semantics ---
        t, j, _ = vitpose_positions_2d.shape
        vitpose_map = {name: i for i, name in enumerate(VITPOSE_25_NAMES)}

        # Get indices of base joints needed for virtual joint calculation
        lhip_idx, rhip_idx = vitpose_map['left_hip'], vitpose_map['right_hip']
        lshoulder_idx, rshoulder_idx = vitpose_map['left_shoulder'], vitpose_map['right_shoulder']

        # Calculate virtual joints for all frames at once
        root_joints = (vitpose_positions_2d[:, lhip_idx, :] + vitpose_positions_2d[:, rhip_idx, :]) / 2.0
        chest_joints = (vitpose_positions_2d[:, lshoulder_idx, :] + vitpose_positions_2d[:, rshoulder_idx, :]) / 2.0

        # Build the final joint name list and a map for easy lookup
        self.joint_names = SMPL_NAMES_NEEDED
        self.joint_map = {name: i for i, name in enumerate(self.joint_names)}

        # Create a new position array with the extended joints
        num_new_joints = len(self.joint_names)
        self.positions = np.zeros((t, num_new_joints, 2), dtype=np.float32)

        # --- Fill the new positions array by mapping names ---
        for smpl_name, smpl_idx in self.joint_map.items():
            # Handle virtual joints
            if smpl_name in ["root", "belly", "spine"]:
                self.positions[:, smpl_idx, :] = root_joints
            elif smpl_name in ["chest", "neck"]:
                self.positions[:, smpl_idx, :] = chest_joints
            # Handle direct mappings and substitutions
            else:
                mapping = {
                    "lhip": "left_hip", "rhip": "right_hip",
                    "lknee": "left_knee", "rknee": "right_knee",
                    "lankle": "left_ankle", "rankle": "right_ankle",
                    "ltoes": "left_big_toe", "rtoes": "right_big_toe",
                    "lshoulder": "left_shoulder", "rshoulder": "right_shoulder",
                    "linshoulder": "left_shoulder", "rinshoulder": "right_shoulder",
                    "lelbow": "left_elbow", "relbow": "right_elbow",
                    "lwrist": "left_wrist", "rwrist": "right_wrist",
                    "lhand": "left_wrist", "rhand": "right_wrist",
                    "head": "nose",
                }
                vitpose_name = mapping.get(smpl_name)
                if vitpose_name and vitpose_name in vitpose_map:
                    vitpose_idx = vitpose_map[vitpose_name]
                    self.positions[:, smpl_idx, :] = vitpose_positions_2d[:, vitpose_idx, :]

        # --- Dynamic Normalization (same as before, but using the new positions array) ---
        lshoulder_pos = self.positions[0, self.joint_map["lshoulder"], :]
        rshoulder_pos = self.positions[0, self.joint_map["rshoulder"], :]
        self.sw = np.linalg.norm(lshoulder_pos - rshoulder_pos) + 1e-6
        self.hw = np.linalg.norm(self.positions[0, self.joint_map["lhip"], :] - self.positions[0, self.joint_map["rhip"], :]) + 1e-6
        self.hl = self.sw  # Use shoulder width as general length reference

    def next_frame(self):
        self.frame_num += 1

    def transform_and_fetch_position(self, j):
        # This becomes much simpler now!
        if j == "y_unit": return np.array([0, 1])
        if j == "minus_y_unit": return np.array([0, -1])
        if j == "zero": return np.array([0, 0])
        if j == "y_min":
            all_y_coords = self.positions[self.frame_num, :, 1]
            return np.array([0, np.max(all_y_coords) if len(all_y_coords) > 0 else 0])
        # All other joints are looked up by their SMPL name in our pre-processed array
        return self.positions[self.frame_num, self.joint_map[j], :]

    def transform_and_fetch_prev_position(self, j):
        return self.positions[self.frame_num - 1, self.joint_map[j], :]

    # The `f_` methods do not need to be changed at all, as they now work with the
    # pre-processed self.positions array transparently.
    def f_move(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold(j1, j1_prev, j2, j2_prev, j3, j3_prev, range)
    def f_nmove(self, j1, j2, j3, j4, range):
        j1_prev, j2_prev, j3_prev, j4_prev = [self.transform_and_fetch_prev_position(j) for j in [j1, j2, j3, j4]]
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.velocity_direction_above_threshold_normal(j1, j1_prev, j2, j3, j4, j4_prev, range)
    def f_plane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane(j1, j2, j3, j4, threshold)
    def f_nplane(self, j1, j2, j3, j4, threshold):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.distance_from_plane_normal(j1, j2, j3, j4, threshold)
    def f_angle(self, j1, j2, j3, j4, range):
        j1, j2, j3, j4 = [self.transform_and_fetch_position(j) for j in [j1, j2, j3, j4]]
        return feat_utils.angle_within_range(j1, j2, j3, j4, range)
    def f_fast(self, j1, threshold):
        j1_prev = self.transform_and_fetch_prev_position(j1)
        j1 = self.transform_and_fetch_position(j1)
        return feat_utils.velocity_above_threshold(j1, j1_prev, threshold)


# [VITPOSE ADAPTATION]
# The main function to call, now using the new class
def geometric_features(joints):
    """
    joints: (b, t, 25, 3) torch.Tensor from ViTPose where last dim is (x, y, confidence)
    returns: (b, 32) torch.Tensor
    """
    b, t, j, _ = joints.shape
    assert j == 25, f"Expected 25 joints from ViTPose, but got {j}"

    # Select only x, y coordinates
    joints_2d = joints[..., :2]
    
    # Root-centering is now handled inside the GeometricFeaturesViTPose class
    # based on the virtual 'root' joint.
    
    joints_np = joints_2d

    geometric_feats = []
    for batch_idx in range(b):
        # Use the new ViTPose-specific class
        f = GeometricFeaturesViTPose(joints_np[batch_idx])

        frame_features = []
        if t <= 1:
            geometric_feats.append(np.zeros(32, dtype=np.float32))
            continue
            
        for frame_idx in range(1, t):
            pose_features = []
            # The feature extraction logic remains IDENTICAL, because the class `f`
            # now handles the translation from SMPL names to ViTPose data.
            pose_features.append(f.f_nmove("neck", "rhip", "lhip", "rwrist", 1.8 * f.hl))
            pose_features.append(f.f_nmove("neck", "lhip", "rhip", "lwrist", 1.8 * f.hl))
            pose_features.append(f.f_nplane("chest", "neck", "neck", "rwrist", 0.2 * f.hl))
            pose_features.append(f.f_nplane("chest", "neck", "neck", "lwrist", 0.2 * f.hl))
            pose_features.append(f.f_move("belly", "chest", "chest", "rwrist", 1.8 * f.hl))
            pose_features.append(f.f_move("belly", "chest", "chest", "lwrist", 1.8 * f.hl))
            pose_features.append(f.f_angle("relbow", "rshoulder", "relbow", "rwrist", [0, 110]))
            pose_features.append(f.f_angle("lelbow", "lshoulder", "lelbow", "lwrist", [0, 110]))
            pose_features.append(f.f_nplane("lshoulder", "rshoulder", "lwrist", "rwrist", 2.5 * f.sw))
            pose_features.append(f.f_move("lwrist", "rwrist", "rwrist", "lwrist", 1.4 * f.hl))
            pose_features.append(f.f_move("rwrist", "root", "lwrist", "root", 1.4 * f.hl))
            pose_features.append(f.f_move("lwrist", "root", "rwrist", "root", 1.4 * f.hl))
            pose_features.append(f.f_fast("rwrist", 2.5 * f.hl))
            pose_features.append(f.f_fast("lwrist", 2.5 * f.hl))
            pose_features.append(f.f_plane("root", "lhip", "ltoes", "rankle", 0.38 * f.hl))
            pose_features.append(f.f_plane("root", "rhip", "rtoes", "lankle", 0.38 * f.hl))
            pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "rankle", 1.2 * f.hl))
            pose_features.append(f.f_nplane("zero", "y_unit", "y_min", "lankle", 1.2 * f.hl))
            pose_features.append(f.f_nplane("lhip", "rhip", "lankle", "rankle", 2.1 * f.hw))
            pose_features.append(f.f_angle("rknee", "rhip", "rknee", "rankle", [0, 110]))
            pose_features.append(f.f_angle("lknee", "lhip", "lknee", "lankle", [0, 110]))
            pose_features.append(f.f_fast("rankle", 2.5 * f.hl))
            pose_features.append(f.f_fast("lankle", 2.5 * f.hl))
            pose_features.append(f.f_angle("neck", "root", "rshoulder", "relbow", [25, 180]))
            pose_features.append(f.f_angle("neck", "root", "lshoulder", "lelbow", [25, 180]))
            pose_features.append(f.f_angle("neck", "root", "rhip", "rknee", [50, 180]))
            pose_features.append(f.f_angle("neck", "root", "lhip", "lknee", [50, 180]))
            pose_features.append(f.f_plane("rankle", "neck", "lankle", "root", 0.5 * f.hl))
            pose_features.append(f.f_angle("neck", "root", "zero", "y_unit", [70, 110]))
            pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "rwrist", -1.2 * f.hl))
            pose_features.append(f.f_nplane("zero", "minus_y_unit", "y_min", "lwrist", -1.2 * f.hl))
            pose_features.append(f.f_fast("root", 2.3 * f.hl))

            frame_features.append(pose_features)
            f.next_frame()

        features = np.mean(np.array(frame_features, dtype=np.float32), axis=0)
        geometric_feats.append(features)

    geometric_feats = np.stack(geometric_feats, axis=0)
    return torch.from_numpy(geometric_feats).float()

