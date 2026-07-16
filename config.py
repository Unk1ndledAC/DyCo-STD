# ComSIA Configuration
# Matching the training protocol described in the paper (Sec. 3.3)

# Optimization
base_lr = 0.001          # Initial learning rate (SGD)
momentum = 0.9           # SGD momentum
weight_decay = 0.0001    # L2 weight decay

# Learning rate schedule (cosine annealing)
T_max = 100              # Total epochs
eta_min = 0.0            # Minimum LR after annealing

# Dynamic Convolution
K = 4                    # Number of parallel convolution kernels
reduction = 4            # Channel reduction ratio in FC layers

# FPN channel dimensions (typical ResNet-50/101 backbone)
fpn_channels = (512, 1024, 2048)  # C3, C4, C5

# Adaptive BBox Regression Loss
smooth_l1_beta = 1.0     # Smooth L1 transition threshold
