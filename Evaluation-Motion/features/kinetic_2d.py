# BSD License (original license retained)
# ... (rest of the license) ...
# Modified by Ruilong Li
# Adapted for 2D ViTPose data by an AI assistant

import numpy as np
import torch
from scipy.signal import convolve

# [VITPOSE ADAPTATION]
# Define the joint set from ViTPose (25 joints) for index mapping
VITPOSE_25_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',           # 0-4
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',     # 5-8
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',               # 9-12
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',             # 13-16
    'head_top', 'neck_viz',                                            # 17-18
    'left_big_toe', 'left_small_toe', 'left_heel',                      # 19-21
    'right_big_toe', 'right_small_toe', 'right_heel',                   # 22-24
]

# [VITPOSE ADAPTATION]
def kinetic_features(joints, frame_time=1/30., sliding_window=2):
    """
    Computes kinetic features for 2D ViTPose joint data.
    
    Args:
        joints (torch.Tensor): A tensor of shape (b, t, 25, 3) from ViTPose,
                               where the last dimension is (x, y, confidence).
        frame_time (float): The time elapsed between frames (e.g., 1/30 for 30 FPS).
        sliding_window (int): The half-size of the smoothing window.

    Returns:
        torch.Tensor: A tensor of shape (b, 75) containing the kinetic features.
    """
    b, t, j, _ = joints.shape
    assert j == 25, f"Expected 25 joints from ViTPose, but got {j}"

    # --- 1. Data Preparation: Convert to 2D and Normalize ---
    # Extract only x, y coordinates
    joints_2d = joints[..., :2]
    
    # Explicitly calculate a stable root (hip center) for normalization
    lhip_idx = VITPOSE_25_NAMES.index('left_hip')
    rhip_idx = VITPOSE_25_NAMES.index('right_hip')
    root_pos = (joints_2d[:, :, lhip_idx:lhip_idx+1, :] + joints_2d[:, :, rhip_idx:rhip_idx+1, :]) / 2.0
    joints_2d = joints_2d - root_pos

    # --- 2. Compute Velocities ---
    # Velocity is calculated in pixels/second
    velocities = (joints_2d[:, 1:, :, :] - joints_2d[:, :-1, :, :]) / frame_time  # Shape: (b, t-1, j, 2)
    # Pad to match original time dimension, making first frame's velocity a copy of the second's
    velocities = np.pad(velocities, ((0,0), (1,0), (0,0), (0,0)), mode="edge")  # Shape: (b, t, j, 2)

    # --- 3. Split into Horizontal and Vertical Components (2D Adaptation) ---
    # In 2D image coordinates, 'horizontal' is the x-axis (index 0) and 'vertical' is the y-axis (index 1)
    horizontal_velocities = velocities[:, :, :, 0:1]  # Shape: (b, t, j, 1)
    vertical_velocities = velocities[:, :, :, 1:2]    # Shape: (b, t, j, 1)

    # --- 4. Sliding Window Smoothing ---
    def sliding_average(arr, window_size):
        if window_size == 0: return arr
        kernel = np.ones(window_size*2+1) / (window_size*2+1)
        arr_smooth = np.zeros_like(arr)
        # Convolve over the time axis (axis=1)
        for b_idx in range(arr.shape[0]):
            for j_idx in range(arr.shape[2]):
                for d_idx in range(arr.shape[3]):
                    arr_smooth[b_idx, :, j_idx, d_idx] = convolve(arr[b_idx, :, j_idx, d_idx], kernel, mode='same', method='auto')
        return arr_smooth

    velocities = sliding_average(velocities, sliding_window)
    horizontal_velocities = sliding_average(horizontal_velocities, sliding_window)
    vertical_velocities = sliding_average(vertical_velocities, sliding_window)

    # --- 5. Compute Kinetic Energy Features ---
    # These are proportional to kinetic energy (mass is assumed constant and ignored)
    # We sum the squared velocities over the component dimension (which is just 1) and average over time.
    kinetic_energy_h = np.mean(np.sum(horizontal_velocities**2, axis=-1), axis=1)  # Shape: (b, j)
    kinetic_energy_v = np.mean(np.sum(vertical_velocities**2, axis=-1), axis=1)    # Shape: (b, j)

    # --- 6. Compute Energy Expenditure Feature (Acceleration-based) ---
    # Approximate acceleration in pixels/second^2
    accelerations = (velocities[:, 1:, :, :] - velocities[:, :-1, :, :]) / frame_time
    accelerations = np.pad(accelerations, ((0,0), (1,0), (0,0), (0,0)), mode="edge")
    accelerations = sliding_average(accelerations, sliding_window)
    
    # Energy expenditure is approximated by the average magnitude of acceleration
    # np.linalg.norm on the last axis calculates sqrt(ax^2 + ay^2) for our 2D data
    energy_expenditure = np.mean(np.linalg.norm(accelerations, axis=-1), axis=1)  # Shape: (b, j)

    # --- 7. Stack Features for All Joints ---
    kinetic_feats = []
    for i in range(j):  # Iterate through all 25 joints
        # For each joint, stack its 3 kinetic features
        feat = np.stack([
            kinetic_energy_h[:, i],
            kinetic_energy_v[:, i],
            energy_expenditure[:, i]
        ], axis=-1)  # Shape: (b, 3)
        kinetic_feats.append(feat)

    # Concatenate features from all joints
    kinetic_feats = np.concatenate(kinetic_feats, axis=-1)  # Shape: (b, 75)
    
    # Convert to torch tensor
    return torch.from_numpy(kinetic_feats.astype(np.float32))