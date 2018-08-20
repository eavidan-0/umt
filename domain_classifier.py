import torch.nn as nn
from audio_data import *


class DomainClassifier(nn.Module):
    def __init__(self, classes, bias=True):
        super(DomainClassifier, self).__init__()

        self.classes = classes
        channels_1 = 2 ** (len(DOMAINS) + 1)
        channels_2 = channels_1 //  2

        self.conv_1 = nn.Conv1d(in_channels=classes,
                                out_channels=channels_1,
                                kernel_size=2,
                                bias=bias)

        self.conv_2 = nn.Conv1d(in_channels=channels_1,
                                out_channels=channels_2,
                                kernel_size=2,
                                bias=bias)

        self.conv_3 = nn.Conv1d(in_channels=channels_2,
                                out_channels=len(DOMAINS),
                                kernel_size=2,
                                bias=bias)

    def forward(self, latent):
        x = latent

        x = self.conv_1(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_2(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_3(x)
        x = F.elu(x, alpha=0.7)

        x = F.avg_pool1d(x, kernel_size=x.size(2))
        x = x.squeeze()  # [batch, D]

        return x
