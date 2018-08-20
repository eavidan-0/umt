import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, classes, bias=True):
        super(DomainClassifier, self).__init__()

        self.classes = classes
        channels = classes // 8

        self.conv_1 = nn.Conv1d(in_channels=classes,
                                out_channels=channels,
                                kernel_size=3,
                                bias=bias)

        self.conv_2 = nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=3,
                                bias=bias)

        self.conv_3 = nn.Conv1d(in_channels=channels,
                                out_channels=len(DOMAINS),
                                kernel_size=3,
                                bias=bias)

    def forward(self, latent):
        x = latent

        x = self.conv_1(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_2(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_3(x)
        x = F.elu(x, alpha=1.0)

        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        x = x.squeeze()  # [batch, D]
        x = F.normalize(x, dim=1)

        return x
