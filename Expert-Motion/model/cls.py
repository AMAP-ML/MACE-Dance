import math
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

import yaml
from types import SimpleNamespace
from einops import rearrange
# from utils.utils import denormalize, normalize, similarity_matrix

def normalize(data, mean=0, std=1):
    # std[std < 1e-5] = 1e-5
    return (data - mean) / std

def denormalize(data, mean=0, std=1):
    # std[std < 1e-5] = 1e-5
    return data * std + mean 
    
class CLS(nn.Module):
    def __init__(self, device=None):
        super().__init__()

        self.dance_encoder = Dance_Encoder()
        self.genreLoss = nn.CrossEntropyLoss()
    
        mean, std = torch.load('./Pretrained/cls/mean.pt'), torch.load('./Pretrained/cls/std.pt')
        self.smpl_trans_mean, self.smpl_poses_mean, self.smpl_root_vel_mean, self.music_librosa_mean, self.music_muq_mean  = \
            mean['smpl_trans_mean'].to(device).float(), mean['smpl_poses_mean'].to(device).float(), \
            mean['smpl_root_vel_mean'].to(device).float(), mean['music_librosa_mean'].to(device).float(), mean['music_muq_mean'].to(device).float()
        self.smpl_trans_std, self.smpl_poses_std, self.smpl_root_vel_std, self.music_librosa_std, self.music_muq_std = \
            std['smpl_trans_std'].to(device).float(), std['smpl_poses_std'].to(device).float(), \
            std['smpl_root_vel_std'].to(device).float(), std['music_librosa_std'].to(device).float(), std['music_muq_std'].to(device).float()


    def forward(self, data, device):
        
        smpl_trans = data['smpl_trans'].to(device).float()
        smpl_poses = data['smpl_poses'].to(device).float()
        label_gt = data['label'].to(device).long()

        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)

        label_pred = self.dance_encoder(torch.cat([smpl_trans, smpl_poses], dim=2))
        cls_loss = self.genreLoss(label_pred, label_gt)

        result = {'dance_pred': label_pred, 'dance_gt': label_gt}
        loss = {'total': cls_loss}

        return result, loss


    def inference(self, data, device):
        smpl_trans = data['smpl_trans'].to(device).float()
        smpl_poses = data['smpl_poses'].to(device).float()
        label_gt = data['label'].to(device).long()

        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)

        label_pred = self.dance_encoder(torch.cat([smpl_trans, smpl_poses], dim=2))
        cls_loss = self.genreLoss(label_pred, label_gt)

        result = {'dance_pred': label_pred, 'dance_gt': label_gt}
        loss = {'total': cls_loss}
        return result, loss


    def dance_encode(self, smpl_trans, smpl_poses):
        smpl_trans = normalize(smpl_trans, self.smpl_trans_mean, self.smpl_trans_std)
        smpl_poses = normalize(smpl_poses, self.smpl_poses_mean, self.smpl_poses_std)
        dance_label = self.dance_encoder(torch.cat([smpl_trans, smpl_poses], dim=2))
        return dance_label



class Dance_Encoder(nn.Module):
    def __init__(self, input_size1=147, input_size2=72, hidden_size=512, num_heads=8, num_layer=8):
        super(Dance_Encoder, self).__init__()
        self.MLP_input = nn.Sequential(
            nn.Linear(input_size1, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.MLP_keypoint = nn.Sequential(
            nn.Linear(input_size2, hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        self.MLP_output = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Linear(128, 20)
        )

        blocks = nn.Sequential()
        for i in range(num_layer):
            blocks.add_module('Dance_Former_{}'.format(i), TTransformer(hidden_size=hidden_size, num_heads=num_heads))
            blocks.add_module('Dance_DownSample_{}'.format(i), DownSample(rate=2))
        self.dance_encoder = blocks

    def forward(self, x):
        x = self.MLP_input(x)
        x = self.dance_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.MLP_output(x)
        return x  


class TTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super(TTransformer, self).__init__()
        Layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=0.25)
        self.layer = nn.TransformerEncoder(Layer, num_layers=1)

    def forward(self, x):
        x = rearrange(x, 'b t c -> t b c')
        x = self.layer(x)
        x = rearrange(x, 't b c -> b t c')
        return x

class DownSample(nn.Module):
    def __init__(self, rate=2, hidden_size=512):
        super(DownSample, self).__init__()
        self.pool = nn.AvgPool1d(kernel_size=3, stride=rate, padding=1)

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.pool(x)
        x = rearrange(x, 'b c t -> b t c')
        return x