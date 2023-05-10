import torch.nn as nn
import torchvision.models as models


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        return x


