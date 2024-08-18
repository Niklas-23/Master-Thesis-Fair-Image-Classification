import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FairSupConResNet18(nn.Module):
    """backbone + projection head"""

    def __init__(self, feat_dim=128):
        super().__init__()
        self.resnet_conv = nn.Sequential(
            *list(models.resnet18(weights=None).children())[:-2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )

    def forward(self, x):
        x = self.resnet_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = F.normalize(self.head(x), dim=1)
        return out

    def encoder(self, x):
        x = self.resnet_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_intermediate_feature_representations(self, x):
        return self.encoder(x)


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=2, feat_dim=512):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
