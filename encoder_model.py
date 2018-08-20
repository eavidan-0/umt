import os
import os.path
import time
from wavenet_modules import *
from audio_data import *
import math
import numpy as np

# Umt downsampled by x12.5, but need divisability
# Nsynth used x32
DOWNSAMPLE_FACTOR = 40
ENC_LEN = SR / DOWNSAMPLE_FACTOR
if not ENC_LEN == int(ENC_LEN):
    raise ValueError("SR not divisable") 

ENC_LEN = int(ENC_LEN)

class EncoderModel(nn.Module):
    def __init__(self,
                 classes,
                 blocks=3,
                 layers=10,
                 channels=128,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=True):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        prev_dilation = 1

        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.classes,
                                    out_channels=channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            dilation = 1
            for i in range(layers):
                # dilations of this layer - padding in order to keep constant channel width
                padding = math.ceil(dilation * (self.kernel_size - 1) / 2)
                self.dilated_convs.append(nn.Conv1d(in_channels=channels,
                                                    out_channels=channels,
                                                    kernel_size=self.kernel_size,
                                                    dilation=dilation,
                                                    padding=padding,
                                                    bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=channels,
                                                     out_channels=channels,
                                                     kernel_size=1,
                                                     bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                prev_dilation = dilation
                dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=channels,
                                  out_channels=classes,
                                  kernel_size=1,
                                  bias=True)

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
            start_idx = 0 if x.size() == residual.size() else (self.kernel_size - 1)
            x = x + residual[:, :, start_idx:]

        x = self.end_conv(x)
        latent = F.avg_pool1d(x, kernel_size=DOWNSAMPLE_FACTOR)

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
        # for q in self.dilated_queues:
        #     q.dtype = type

        for c in self.dilated_convs:
            c.cuda(device)

        for c in self.residual_convs:
            c.cuda(device)

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
