import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDueling(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDueling, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=10, stride=4),  # Input channel = 3: there are 3 channels for image (rgb)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=10, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=10, stride=1),
            nn.ReLU(True)
        )

        self.value = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(4608, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = torch.unsqueeze(x, dim=1)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)

        # Dueline DQN
        val = self.value(x)
        adv = self.advantage(x)
        avg = torch.mean(adv, dim=1, keepdim=True)
        return val + adv - avg

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

