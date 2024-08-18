import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()

        resnet = models.resnet18(weights=None)

        # Create new last layer
        in_features = resnet.fc.in_features
        updated_fc = nn.Linear(in_features, num_classes)

        # Initialize new fully connected layer
        nn.init.kaiming_normal_(updated_fc.weight)
        nn.init.zeros_(updated_fc.bias)

        # Update resnet fully connected layer
        resnet.fc = updated_fc
        self.resnet = resnet

    def forward(self, x, get_inter=False):
        if get_inter:
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            b1 = self.resnet.layer1(x)
            b2 = self.resnet.layer2(b1)
            b3 = self.resnet.layer3(b2)
            b4 = self.resnet.layer4(b3)

            x = self.resnet.avgpool(b4)
            x = torch.flatten(x, 1)
            x = self.resnet.fc(x)

            return b1, b2, b3, b4, x
        else:
            return self.resnet(x)

    def get_intermediate_feature_representations(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        b1 = self.resnet.layer1(x)
        b2 = self.resnet.layer2(b1)
        b3 = self.resnet.layer3(b2)
        b4 = self.resnet.layer4(b3)
        x = self.resnet.avgpool(b4)
        return torch.flatten(x, 1)
