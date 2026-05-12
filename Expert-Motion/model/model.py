from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from model.utils import SinusoidalPosEmb
from model.rotary_embedding_torch import RotaryEmbedding
from mamba_ssm import Mamba


class DenseFiLM(nn.Module):
    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(nn.Mish(), nn.Linear(embed_channels, embed_channels * 2))

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class IntraModalMamba(nn.Module):
    def __init__(self, d_model, dropout, layer_norm_eps,):
        super().__init__()
        self.mamba_forward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_backward = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.mamba_forward(x)
        x2 = self.mamba_backward(x.flip(1)).flip(1)
        x = self.dropout(x1+x2)
        return x

class CrossModalTransformer(nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout, batch_first, layer_norm_eps, activation):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm_dance = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_music = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, mem):
        x = self.dropout_cross(self.cross_attn(self.norm_dance(x), self.norm_music(mem), self.norm_music(mem), need_weights=False, )[0])
        x = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(self.norm(x))))))
        return x


            
class DanceFiLMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True):
        super().__init__()
        self.bimamba = IntraModalMamba(d_model, dropout, layer_norm_eps)
        self.transformer = CrossModalTransformer(d_model, dim_feedforward, nhead, dropout, batch_first, layer_norm_eps, activation)

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)

    # x, cond, t
    def forward(self, x, cond, genre, t):
        # intra-modal -> film -> residual
        x_1 = self.bimamba(x)
        x = x + featurewise_affine(x_1, self.film1(t))

        # cross-modal -> film -> residual
        x_2 = self.transformer(x, cond)
        x = x + featurewise_affine(x_2, self.film2(t))
        return x



class MusicFiLMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True):
        super().__init__()
        self.bimamba = IntraModalMamba(d_model, dropout, layer_norm_eps)
        self.film = DenseFiLM(d_model)
        self.genre_MLP = nn.Sequential(nn.Embedding(16, d_model), nn.Linear(d_model, dim_feedforward), nn.Mish(), nn.Linear(dim_feedforward, d_model))

    def forward(self, x, genre):
        x_1 = self.bimamba(x)
        genre = self.genre_MLP(genre)
        x = x + featurewise_affine(x_1, self.film(genre))
        return x


class DanceDecoder(nn.Module):
    def __init__(self, nfeats, seq_len=150, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4,
                 dropout=0.1, cond_feature_dim=4800, activation=F.gelu, use_rotary=True,  **kwargs):
        super().__init__()
        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        # beta embedding
        self.beta_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
            nn.Linear(latent_dim * 4, latent_dim),
        )

        # input and condition projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.final_layer = nn.Linear(latent_dim, nfeats)

        # Dance and Music stacks
        self.dance_num_layers = 8
        self.dancelayerstack = nn.ModuleList([
            DanceFiLMLayer(latent_dim, num_heads, ff_size, dropout, activation, batch_first=True)
            for _ in range(self.dance_num_layers)
        ])

        self.music_num_layers = 2
        self.musiclayerstack = nn.ModuleList([
            MusicFiLMLayer(latent_dim, num_heads, ff_size, dropout, activation, batch_first=True)
            for _ in range(self.music_num_layers)
        ])

        self.null_cond_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))

    def forward(self, x, cond_embed, genre, times, beta, cond_drop_prob=0.2):
        batch_size, device = x.shape[0], x.device

        # latent projection
        x = self.input_projection(x)
        # combine time + beta embeddings
        t = self.time_mlp(times) + self.beta_mlp(beta * 1000)

        cond = self.cond_projection(cond_embed)
        # Music encoder
        for i in range(self.music_num_layers):
            cond = self.musiclayerstack[i](cond, genre)

        mask = (torch.rand(batch_size, device=device) > cond_drop_prob).float().view(batch_size, 1, 1)
        cond = cond * mask + self.null_cond_embed.to(device)

        # Dance decoder
        for i in range(self.dance_num_layers):
            x = self.dancelayerstack[i](x, cond, genre, t)

        output = self.final_layer(x)
        return output

    def train_pred(self, x, cond, genre, times):
        batch_size, device = x.shape[0], x.device
        beta = torch.rand(batch_size, device=device)

        # conditional pred
        cond_pred = self.forward(x, cond, genre, times, beta, cond_drop_prob=0.0)
        # unconditional pred (detach gradients)
        uncond_pred = self.forward(x, cond, genre, times, beta, cond_drop_prob=1.0)
        uncond_pred = uncond_pred.detach()

        pred = beta.view(batch_size, 1, 1) * cond_pred + (1.0 - beta.view(batch_size, 1, 1)) * uncond_pred
        return pred

    @torch.no_grad()
    def infer_pred(self, x, cond, genre, times, beta=0.75):
        batch_size, device = x.shape[0], x.device
        beta = torch.full((batch_size, 1), beta, device=device).squeeze()
        pred = self.forward(x, cond, genre, times, beta, cond_drop_prob=0.0)
        return pred
 



