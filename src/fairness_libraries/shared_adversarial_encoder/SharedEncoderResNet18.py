import torch
import torch.nn as nn
import torchvision.models as models


class SharedAdversarialEncoderResNet18(nn.Module):
    def __init__(self, num_classes_target=2, feat_dim=512, num_classes_protected_features=2):
        super(SharedAdversarialEncoderResNet18, self).__init__()

        resnet = models.resnet18(weights="IMAGENET1K_V1")

        # Create new last layer
        in_features = resnet.fc.in_features
        updated_fc = nn.Linear(in_features, feat_dim)

        # Initialize new fully connected layer
        nn.init.kaiming_normal_(updated_fc.weight)
        nn.init.zeros_(updated_fc.bias)

        # Update resnet fully connected layer
        resnet.fc = updated_fc
        self.resnet = resnet

        for param in self.resnet.parameters():
            param.requires_grad = True

        self.bn = nn.BatchNorm1d(num_features=feat_dim, momentum=0.01)

        self.classifier_head = torch.nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes_target)
        )

        self.adversarial_head = torch.nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=64, out_features=num_classes_protected_features)
        )

    def forward(self, x, get_adv=False):
        if get_adv:
            h = self.resnet(x)
            y = self.classifier_head(h)
            a = self.adversarial_head(h)
            a_detach = self.adversarial_head(h.detach())
            return y, a, a_detach
        else:
            return self.classifier_head(self.resnet(x))

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
