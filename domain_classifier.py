import torch.nn as nn
from audio_data import *


class DomainClassifier(nn.Module):
    def __init__(self, classes, bias=True):
        super(DomainClassifier, self).__init__()

        self.classes = classes

        self.conv_1 = nn.Conv1d(in_channels=classes,
                                out_channels=classes,
                                kernel_size=8,
                                bias=bias)

        self.conv_2 = nn.Conv1d(in_channels=classes,
                                out_channels=classes,
                                kernel_size=8,
                                bias=bias)

        self.conv_3 = nn.Conv1d(in_channels=classes,
                                out_channels=len(DOMAINS),
                                kernel_size=8,
                                bias=bias)

    def forward(self, latent):
        x = latent

        x = self.conv_1(x)
        x = F.elu(x, alpha=0.9)
        x = self.conv_2(x)
        x = F.elu(x, alpha=0.8)
        x = self.conv_3(x)
        x = F.elu(x, alpha=0.7)

        x = F.avg_pool1d(x, kernel_size=x.size(2))
        x = x.squeeze()  # [batch, D]

        return x
