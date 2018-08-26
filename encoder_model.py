import os
import os.path
import time
from wavenet_modules import *
from audio_data import *
import math
import numpy as np

ENC_LEN = 64
POOL_KERNEL = 512


class EncoderModel(nn.Module):
    def __init__(self,
                 classes,
                 bias,
                 blocks,
                 layers,
                 channels=128,
                 kernel_size=2,
                 dtype=torch.FloatTensor):

        super(EncoderModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.classes = classes
        self.dtype = dtype

        # build model
        input_trim = 0

        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilated_convs.append(nn.Conv1d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=kernel_size,
                                                    dilation=dilation,
                                                    bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=channels,
                                                     out_channels=channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # increase kernel size
                input_trim += dilation * (kernel_size - 1)
                dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=channels,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True)

        self.input_trim = input_trim

    def forward(self, input):
        x = self.start_conv(input)

        # # WaveNet layers
        for i in range(self.blocks * self.layers):
            # Step 1: ReLUs
            residual = F.relu(x)

            # Step 2: dilated convolution
            residual = self.dilated_convs[i](residual)

            # Step 3: ReLU
            residual = F.relu(residual)

            # Step 4: Just a 1x1 convolution
            residual = self.residual_convs[i](residual)

            # Step 5: Skip and Residual summation
            layer_output_length = residual.size(2)
            x = x[:, :, -layer_output_length:] + residual

        output_length = input.size()[2] - self.input_trim
        assert x.size()[2] == output_length, "expected encoder output %d, got %r" % (
            output_length, x.size())

        x = self.end_conv(x)
        
        # TODO: wtf?
        # latent = F.avg_pool1d(x, kernel_size=32)
        latent = F.avg_pool1d(x, kernel_size=POOL_KERNEL)

        # assert latent.size()[2] == ENC_LEN, "expected latent %d, got %r" % (
            # ENC_LEN, latent.size())
        return latent

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type):
        self.dtype = type
        super().cpu()

    def cuda(self, device, type):
        self.dtype = type
        super().cuda(device)
