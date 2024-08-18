import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class ResNet18FCRO(nn.Module):
    def __init__(self, num_classes=2, representation_embedding_dim=128):
        super().__init__()
        resnet = models.resnet18(pretrained=False)

        # Create new layers
        in_features = resnet.fc.in_features
        self.fc = nn.Linear(in_features, representation_embedding_dim)
        self.out = nn.Linear(representation_embedding_dim, num_classes)

        # Initialize new fully connected layer
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.kaiming_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Update resnet fully connected layer
        resnet.fc = self.fc
        self.resnet = resnet

    def forward(self, x):
        feature_embedding = self.resnet(x)
        feature_embedding = F.normalize(feature_embedding, dim=-1)
        out = self.out(feature_embedding)

        return out, feature_embedding

    def get_intermediate_feature_representations(self, x):
        return self.resnet(x)
