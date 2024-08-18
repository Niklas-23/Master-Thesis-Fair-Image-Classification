import torch
import torch.nn as nn

import torchvision.models as models

from src.fairness_libraries.entangling_disentangling_bias.EnD import pattern_norm


class ResNet18EnD(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet_conv = nn.Sequential(
            *list(models.resnet18(weights=None).children())[:-2]
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            pattern_norm()
        )
        self.fc = nn.Linear(512, num_classes)

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.resnet_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

    def get_intermediate_feature_representations(self, x):
        x = self.resnet_conv(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)
