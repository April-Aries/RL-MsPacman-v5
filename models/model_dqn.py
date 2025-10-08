import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtariNetDQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(AtariNetDQN, self).__init__()
        self.cnn = nn.Sequential(
            # Input channel = 3: there are 3 channels for image (rgb)
            nn.Conv2d(3, 16, kernel_size=30, stride=3),
            # Output: (16, 61, 51)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=20, stride=2),
            # Output: (32, 21, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=10, stride=1),
            # Output: (64, 12, 7)
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(5376, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

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

