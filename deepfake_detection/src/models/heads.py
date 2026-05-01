"""Swappable classification heads."""

import torch.nn as nn


class BinaryHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_model//2, 2))
    def forward(self, x): return self.head(x)


class MultiClassHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(d_model//2, num_classes))
    def forward(self, x): return self.head(x)


class LocalizationHead(nn.Module):
    def __init__(self, d_model, out_size=14):
        super().__init__()
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, out_size * out_size))
    def forward(self, x):
        return self.head(x).reshape(
            x.size(0), 1, self.out_size, self.out_size)
