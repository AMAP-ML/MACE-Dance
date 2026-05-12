# BSD License

# For fairmotion software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Ruilong Li

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
from . import utils as feat_utils
import torch

def kinetic_features(joints, frame_time=1/30., up_vec="y", sliding_window=2):
    """
    joints: (b, t, j, 3)
    output: (b, 72)
    """
    b, t, j, _ = joints.shape
    joints = joints.numpy()
    joints = joints - joints[:, :1, :1, :]  # Normalize by root

    # Precompute velocities
    velocities = (joints[:, 1:, :, :] - joints[:, :-1, :, :]) / frame_time  # (b, t-1, j, 3)
    
    # Pad velocities to match original time dim
    velocities = np.pad(velocities, ((0,0), (1,0), (0,0), (0,0)), mode="edge")  # (b, t, j, 3)

    # Compute horizontal velocities
    if up_vec == "y":
        horizontal_velocities = velocities[:, :, :, [0, 2]]  # x and z
        vertical_velocities = velocities[:, :, :, 1:2]        # y
    elif up_vec == "z":
        horizontal_velocities = velocities[:, :, :, [0, 1]]  # x and y
        vertical_velocities = velocities[:, :, :, 2:3]        # z
    else:
        raise NotImplementedError("up_vec must be 'y' or 'z'.")

    # Sliding window smoothing
    def sliding_average(arr, window_size):
        kernel = np.ones(window_size*2+1) / (window_size*2+1)
        # Apply 1D convolution over time axis
        from scipy.signal import convolve
        arr_smooth = np.zeros_like(arr)
        for b_idx in range(arr.shape[0]):
            for j_idx in range(arr.shape[2]):
                for d_idx in range(arr.shape[3]):
                    arr_smooth[b_idx, :, j_idx, d_idx] = convolve(arr[b_idx, :, j_idx, d_idx], kernel, mode='same')
        return arr_smooth

    velocities = sliding_average(velocities, sliding_window)
    horizontal_velocities = sliding_average(horizontal_velocities, sliding_window)
    vertical_velocities = sliding_average(vertical_velocities, sliding_window)

    # Compute kinetic energy and energy expenditure
    kinetic_energy_h = np.mean(np.sum(horizontal_velocities**2, axis=-1), axis=1)  # (b, j)
    kinetic_energy_v = np.mean(np.sum(vertical_velocities**2, axis=-1), axis=1)    # (b, j)

    # Approximate acceleration
    accelerations = (velocities[:, 1:, :, :] - velocities[:, :-1, :, :]) / frame_time  # (b, t-1, j, 3)
    accelerations = np.pad(accelerations, ((0,0), (1,0), (0,0), (0,0)), mode="edge")  # (b, t, j, 3)
    accelerations = sliding_average(accelerations, sliding_window)
    energy_expenditure = np.mean(np.linalg.norm(accelerations, axis=-1), axis=1)  # (b, j)

    # Stack features
    kinetic_feats = []
    for i in range(j):
        # Each joint: [horizontal_energy, vertical_energy, energy_expenditure]
        feat = np.stack([
            kinetic_energy_h[:, i],
            kinetic_energy_v[:, i],
            energy_expenditure[:, i]
        ], axis=-1)  # (b, 3)
        kinetic_feats.append(feat)

    kinetic_feats = np.concatenate(kinetic_feats, axis=-1)  # (b, 72)
    kinetic_feats = torch.from_numpy(kinetic_feats.astype(np.float32))
    return kinetic_feats[:, :66]

# def kinetic_features(joints):
#     # joints: b t j 3
#     b, t, j, _ = joints.shape
#     joints = joints.cpu().numpy()
#     joints = joints - joints[:, :1, :1, :]
#     kinetic_feats = np.zeros((b, 72))
#     for i in range(b):
#         kinetic_feats[i, :] = extract_kinetic_features(joints[i])
#     return kinetic_feats

def extract_kinetic_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


class KineticFeatures:
    def __init__(
        self, positions, frame_time=1./30, up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += feat_utils.calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val
