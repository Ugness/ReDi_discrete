import os
import math
import torch
import torch.nn as nn
import numpy as np

def GetConv2d(C_in, C_out, kernel_size, stride=1, padding=0):
    return nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False)


def GetLinear(C_in, C_out):
    return nn.Linear(C_in, C_out, bias=False)

# ===== Embedding =====

class Fourier(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        self.register_buffer('frequencies', torch.randn(embedding_size))
        self.register_buffer('phases', torch.rand(embedding_size))

    def forward(self, a):
        b = (2 * np.pi) * (a[:, None] * self.frequencies[None, :] + self.phases[None, :])
        b = torch.cos(b)
        return b


class Embedding(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.fourier = Fourier(embedding_size=n_channels // 4)
        self.linear1 = GetLinear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.linear2 = GetLinear(n_channels, n_channels)

    def forward(self, c_noise):
        emb = self.fourier(c_noise)
        emb = self.act(self.linear1(emb))
        emb = self.act(self.linear2(emb))
        return emb


class Reweighting(nn.Module):
    def __init__(self, n_channels=256):
        super().__init__()
        self.fourier = Fourier(embedding_size=n_channels)
        self.linear = GetLinear(n_channels, 1)

    def forward(self, x, c_noise):
        emb = self.fourier(c_noise)
        emb = self.linear(emb)
        return emb

class Reweighting2(nn.Module):
    def __init__(self, net, n_channels=256):
        super().__init__()
        self.net = net
        self.fourier = Fourier(embedding_size=n_channels)
        self.linear = GetLinear(n_channels, 1)

    def forward(self, x, c_noise):
        t_emb = self.fourier(c_noise).unsqueeze(1)
        x = torch.cat([x, t_emb], dim=1)
        emb, _ = self.net(x)
        emb = self.linear(emb[:, -1])
        return emb
