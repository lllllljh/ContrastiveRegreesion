import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Flatten()

    def forward(self, x):
        x = self.model(x)
        return x
