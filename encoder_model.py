import os
import os.path
import time
from wavenet_modules import *
from audio_data import *
import math
import numpy as np

ENC_LEN = 64
POOL_KERNEL = 800
PRE_POOL_LENGTH = ENC_LEN * POOL_KERNEL


class EncoderModel(nn.Module):
    def __init__(self,
                 classes,
                 blocks=3,
                 layers=10,
                 channels=128,
                 initial_kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=True):

        super(EncoderModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.classes = classes
        self.dtype = dtype

        # build model
        input_trim = 0

        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.kernels = []

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            dilation = 4
            kernel_size = initial_kernel_size
            for i in range(layers):
                # dilations of this layer - padding in order to keep constant channel width
                padding = math.ceil(dilation * (kernel_size - 1) / 2)
                self.kernels.append(kernel_size)
                self.dilated_convs.append(nn.Conv1d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=kernel_size,
                                                    dilation=dilation,
                                                    padding=padding,
                                                    bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=channels,
                                                     out_channels=channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # increase kernel size
                input_trim += dilation * (kernel_size - 1) - 2 * padding
                # dilation *= 2
                kernel_size *= 2

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
            # start_idx overcomes dilated_conv with non-integer padding being rounded
            start_idx = 0 if x.size() == residual.size() else (self.kernels[i]- - 1)
            print("conditioning", x.size(), residual.size(), start_idx)
            x = x + residual[:, :, start_idx:]

        output_length = input.size()[2] - self.input_trim
        print("sizes", x.size(), input.size(), self.input_trim)
        assert x.size()[2] == output_length

        x = self.end_conv(x)
        latent = F.avg_pool1d(x, kernel_size=output_length//ENC_LEN)

        print("post pool", slatent.size())
        return latent

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type):
        self.dtype = type
        # for q in self.dilated_queues:
        #     q.dtype = type

        for c in self.dilated_convs:
            c.cpu()

        for c in self.residual_convs:
            c.cpu()

        super().cpu()

    def cuda(self, device, type):
        self.dtype = type
        super().cuda(device)


def load_latest_model_from(location, use_cuda=True):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)

    if use_cuda:
        model = torch.load(newest_file)
    else:
        model = load_to_cpu(newest_file)

    return model


def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model
