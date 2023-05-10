import torch.nn as nn
import timm


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.model = timm.create_model('xception', pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x
