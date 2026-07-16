"""
Adaptive Bounding Box Regression Loss with SNR-based Sample Weighting.

Implements the loss function described in the paper (Sec. 3.3, Eq. (5)):

  L_reg = (1 / N_pos) * sum_i w'_i * SmoothL1(pred_i, gt_i)

where w'_i is computed from the signal-to-noise ratio (SNR) of the
dynamic convolution output feature map:

  SNR = var_local(box_region) / var_global(feature_map)
  w'  = (1 / SNR)  normalized so sum(w') = N

This ensures that small targets (with lower local variance relative to
global background) receive higher regression weights, addressing the
bottleneck of feature mismatch in small target detection.
"""

import torch
import torch.nn as nn


class AdaptiveBBoxRegressionLoss(nn.Module):
    """
    Smooth L1 loss with SNR-based adaptive per-sample weighting.

    Args:
        beta: Threshold for L1/L2 transition in SmoothL1Loss (default 1.0).
    """

    def __init__(self, beta: float = 1.0):
        super(AdaptiveBBoxRegressionLoss, self).__init__()
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def compute_snr_weights(
        self,
        feature_map: torch.Tensor,
        bbox_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample SNR-based weights.

        SNR = var(local bbox region) / var(global feature map)

        Weights w' = (1/SNR) normalized so sum(w') = N,
        giving higher weight to small/low-SNR targets.

        Args:
            feature_map: Dynamic conv output, shape (N, C, H, W).
            bbox_coords: GT bounding boxes, shape (N, 4) as (x1, y1, x2, y2).

        Returns:
            w_prime: Normalized adaptive weights, shape (N,).
        """
        N, C, H, W = feature_map.shape

        # Global variance per sample
        global_var = feature_map.var(dim=(1, 2, 3), unbiased=False)  # (N,)

        # Local variance within each bbox region
        local_vars = []
        for i in range(N):
            x1, y1, x2, y2 = bbox_coords[i].long()
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(x1 + 1, min(x2, W))
            y2 = max(y1 + 1, min(y2, H))
            local_region = feature_map[i, :, y1:y2, x1:x2]
            local_vars.append(local_region.var(unbiased=False))
        local_var = torch.stack(local_vars)  # (N,)

        # Signal-to-Noise Ratio
        snr = local_var / (global_var + 1e-8)  # (N,)

        # Inverse SNR weights, normalized to sum = N
        w_prime = 1.0 / (snr + 1e-8)
        w_prime = w_prime / (w_prime.sum() + 1e-8) * N

        return w_prime

    def forward(
        self,
        pred_bbox: torch.Tensor,
        gt_bbox: torch.Tensor,
        feature_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute adaptive bounding box regression loss.

        Args:
            pred_bbox:   Predicted bboxes, shape (N, 4).
            gt_bbox:     Ground-truth bboxes, shape (N, 4).
            feature_map: Dynamic conv output feature map, shape (N, C, H, W).

        Returns:
            Scalar loss value.
        """
        # Compute SNR-based sample weights ----- [Eq. (5)]
        w_prime = self.compute_snr_weights(feature_map, gt_bbox)  # (N,)

        # Per-sample Smooth L1, summed over 4 coordinates
        smooth_l1_per_sample = self.smooth_l1(pred_bbox, gt_bbox).sum(dim=-1)  # (N,)

        # Weighted mean
        loss = (w_prime * smooth_l1_per_sample).mean()

        return loss
