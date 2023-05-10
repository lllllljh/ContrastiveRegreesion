import timm
import torch.nn as nn


class InceptionResNetV2(nn.Module):
    def __init__(self):
        super(InceptionResNetV2, self).__init__()
        self.model = timm.create_model('inception_resnet_v2', pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

