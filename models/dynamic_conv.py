"""
Dynamic Convolution Module for Small Target Detection in Remote Sensing Imagery.

Implements the ComSIA method described in the paper:
  "An Improved Algorithm for Small Target Detection in Remote Sensing Imagery
   Based on Dynamic Convolution"

Core mechanism:
  1. Global Average Pooling (GAP) over spatial dims   [Eq. (1)]
  2. Two FC layers with ReLU for attention weights     [Eq. (2)]
  3. Reshape + Softmax for K normalized kernel weights [Eq. (3)]
  4. Weighted fusion of K convolution kernels          [Eq. (4)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DynamicConv2d(nn.Module):
    """
    Dynamic Convolution with K parallel kernels and channel-wise attention.

    Mathematical formulation (matching the paper):
      F_c = GAP(X)                                        -- Eq. (1): channel statistics
      W_raw = FC2(ReLU(FC1(F_c)))                         -- Eq. (2): raw attention weights
      attn_k = softmax_k( pool_{C->K}(W_raw) )            -- Eq. (3): normalized K weights
      Y = sum_{k=1}^{K} attn_k * Conv_k(X)                -- Eq. (4): weighted kernel fusion

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Spatial size of convolution kernels (default 3).
        stride:       Convolution stride.
        padding:      Convolution padding.
        groups:       Number of blocked connections.
        bias:         Whether conv layers use bias.
        K:            Number of parallel convolution kernels (default 4).
        reduction:    Channel reduction ratio in FC layers (default 4).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = False,
        K: int = 4,
        reduction: int = 4,
    ):
        super(DynamicConv2d, self).__init__()
        assert in_channels % K == 0, \
            f"in_channels ({in_channels}) must be divisible by K ({K})"

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int)
            else kernel_size
        )

        # K parallel convolution kernels ---------- [Eq. (4)]
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=bias,
            )
            for _ in range(K)
        ])

        # Attention weight generation network ---- [Eq. (1)-(2)]
        # reduced_dim = max(C / r, K)  ensures minimal bottleneck width
        reduced_dim = max(in_channels // reduction, K)

        self.gap = nn.AdaptiveAvgPool2d(1)              # Eq. (1)
        self.fc1 = nn.Linear(in_channels, reduced_dim)  # FC1: C -> C/r
        self.fc2 = nn.Linear(reduced_dim, in_channels)  # FC2: C/r -> C

        # He initialization (matching paper Sec 3.3)
        self._initialize_weights()

    def _initialize_weights(self):
        """He (Kaiming) initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map, shape (B, C, H, W).

        Returns:
            Output feature map after dynamic convolution, shape (B, out_C, H, W).
        """
        B, C, H, W_spatial = x.shape                 # spatial width W_spatial

        # ---- Eq. (1): Channel-wise statistics via GAP ----
        F_c = self.gap(x)                            # (B, C, 1, 1)
        F_c = F_c.view(B, C)                         # (B, C)

        # ---- Eq. (2): FC-ReLU-FC attention generation ----
        raw_weights = self.fc2(F.relu(self.fc1(F_c)))  # (B, C)

        # ---- Eq. (3): Map C-dim to K-dim, then softmax ----
        # raw_weights (B, C) -> reshape (B, K, C/K) -> mean (B, K) -> softmax
        z = raw_weights.view(B, self.K, -1).mean(dim=-1)  # (B, K)
        attn_weights = F.softmax(z, dim=1)                 # (B, K)

        # ---- Eq. (4): Weighted sum of K convolution outputs ----
        y = sum(
            attn_weights[:, i:i + 1, None, None] * self.convs[i](x)
            for i in range(self.K)
        )

        return y


class DynamicConvFPNBlock(nn.Module):
    """
    Wraps DynamicConv2d for insertion into FPN lateral connections.

    Used at a single FPN level (e.g. C3, C4, or C5).
    Keeps in_channels == out_channels to preserve feature dimensions.
    """

    def __init__(self, in_channels: int, K: int = 4, reduction: int = 4):
        super(DynamicConvFPNBlock, self).__init__()
        self.dynamic_conv = DynamicConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            K=K,
            reduction=reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dynamic_conv(x)


class DynamicConvFPN(nn.Module):
    """
    Dynamic Convolution applied to FPN feature pyramid levels C3, C4, C5.

    In the paper (Sec. 3.3), dynamic convolution replaces the standard
    convolution in FPN's bottom-up feature extraction stages C3-C5,
    where C3 (highest resolution) is particularly critical for small
    target detection.

    Training protocol (matching paper):
      - Backbone parameters are frozen
      - Only DynamicConv modules + detection head are fine-tuned
      - SGD with momentum=0.9, weight_decay=1e-4
      - Initial lr=0.001 with cosine annealing
    """

    def __init__(
        self,
        fpn_channels: Tuple[int, int, int] = (512, 1024, 2048),
        K: int = 4,
        reduction: int = 4,
    ):
        super(DynamicConvFPN, self).__init__()

        self.c3_dynamic = DynamicConvFPNBlock(fpn_channels[0], K, reduction)
        self.c4_dynamic = DynamicConvFPNBlock(fpn_channels[1], K, reduction)
        self.c5_dynamic = DynamicConvFPNBlock(fpn_channels[2], K, reduction)

    def forward(
        self,
        c3: torch.Tensor,
        c4: torch.Tensor,
        c5: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply dynamic convolution to each FPN level."""
        c3_out = self.c3_dynamic(c3)
        c4_out = self.c4_dynamic(c4)
        c5_out = self.c5_dynamic(c5)
        return c3_out, c4_out, c5_out
