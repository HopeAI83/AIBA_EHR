
"""PyTorch implementation of the TCN+Attention anomaly detector."""
from __future__ import annotations
import torch
from torch import nn
from config import TCN_DILATION_RATES, TCN_KERNEL_SIZE, TCN_FILTERS, TCN_ATTENTION_HEADS, TCN_DROPOUT

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))

class TCNWithAttention(nn.Module):
    """Dilated causal TCN followed by multi-head temporal attention."""
    def __init__(self, n_features: int, n_filters: int = TCN_FILTERS, dilation_rates=None, kernel_size: int = TCN_KERNEL_SIZE,
                 attention_heads: int = TCN_ATTENTION_HEADS, dropout: float = TCN_DROPOUT):
        super().__init__()
        dilation_rates = dilation_rates or TCN_DILATION_RATES
        layers=[]; in_ch=n_features
        for d in dilation_rates:
            layers.append(TemporalBlock(in_ch, n_filters, kernel_size, d, dropout)); in_ch=n_filters
        self.tcn = nn.Sequential(*layers)
        self.attention = nn.MultiheadAttention(embed_dim=n_filters, num_heads=attention_heads, dropout=dropout, batch_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(n_filters), nn.Linear(n_filters, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))
    def forward(self, x):
        # x: [batch, time, features]
        z = self.tcn(x.transpose(1,2)).transpose(1,2)
        attended, weights = self.attention(z, z, z, need_weights=True)
        pooled = attended.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, weights


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
