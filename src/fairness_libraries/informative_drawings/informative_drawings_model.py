import os

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

norm_layer = nn.InstanceNorm2d

# Code is copied from the informative drawings demo (https://huggingface.co/spaces/carolineec/informativedrawings/blob/main/app.py) to get the minimal code for inference
# Full code base https://github.com/carolineec/informative-drawings

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  nn.ReLU(inplace=True)]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out


class InformativeDrawingsInference:

    def __init__(self):
        current_file_path = os.path.dirname(os.path.abspath(__file__))

        self.model1 = Generator(3, 1, 3)
        self.model1.load_state_dict(torch.load(current_file_path+'/model.pth', map_location=torch.device('cpu')))
        self.model1.eval()

        self.model2 = Generator(3, 1, 3)
        self.model2.load_state_dict(torch.load(current_file_path+'/model2.pth', map_location=torch.device('cpu')))
        self.model2.eval()

    def predict(self, input_img, ver="style 1"):
        input_img = Image.open(input_img)
        # transform = transforms.Compose([transforms.Resize(256, Image.BICUBIC), transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor()])
        input_img = transform(input_img)
        input_img = torch.unsqueeze(input_img, 0)

        with torch.no_grad():
            if ver == 'style 2':
                drawing = self.model2(input_img)[0].detach()
            else:
                drawing = self.model1(input_img)[0].detach()

        drawing = transforms.ToPILImage()(drawing)
        return drawing
