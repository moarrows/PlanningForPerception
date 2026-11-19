
import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3.common.torch_layers import CombinedExtractor


class DepthCNN(nn.Module):

    def __init__(self, features_dim=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(64 * 4 * 4, features_dim),
            nn.ReLU()
        )

        self.features_dim = features_dim

    def forward(self, x):

        # Handle either HWC or CHW image layouts.
        if x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)

        return self.cnn(x.float())


class DroneCombinedExtractor(CombinedExtractor):

    def __init__(self, observation_space):

        # Initially creates the standard SB3 branches.
        super().__init__(
            observation_space,
            cnn_output_dim=256
        )

        # Apply CNN processing to both depth cameras.
        for key in ["depth", "gimbal_depth"]:

            if key not in self.extractors:
                continue

            old_dim = int(np.prod(observation_space[key].shape))

            self.extractors[key] = DepthCNN(
                features_dim=256
            )

            self._features_dim += 256 - old_dim
