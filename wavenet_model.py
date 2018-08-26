import os
import os.path
import time
from wavenet_modules import *
from audio_data import *

import torch.nn.functional as F

from math import ceil


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model

    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        output_length (Int):        Number of samples that are generated for each input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """

    def __init__(self,
                 layers,
                 blocks,
                 dilation_channels,
                 residual_channels,
                 skip_channels,
                 classes=256,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []

        self.cond_convs = nn.ModuleList()
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv_1 = nn.Conv1d(in_channels=classes,
                                      out_channels=residual_channels,
                                      kernel_size=1,
                                      bias=bias)
        
        self.start_conv_2 = nn.Conv1d(in_channels=residual_channels,
                                      out_channels=skip_channels,
                                      kernel_size=1,
                                      bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))
                
                self.cond_convs.append(nn.Conv1d(in_channels=classes,
                                                 out_channels=dilation_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                self.dilated_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                    out_channels=dilation_channels,
                                                    kernel_size=kernel_size,
                                                    dilation=2**i,
                                                    bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        self.end_conv_2 = nn.Conv1d(in_channels=classes,
                                    out_channels=classes,
                                    kernel_size=1,
                                    bias=True)

        self.receptive_field = receptive_field

    def wavenet(self, input_tuple):
        input, en = input_tuple

        # TODO: this was x_scaled, and en was not upsampled
        # l = masked.shift_right(x_scaled)
        l = input
        l = self.start_conv_1(l) 
        s = self.start_conv_2(l) 

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            dilation = 2 ** (i % self.layers)
            d = F.pad(l, ((self.kernel_sze-1)*dilation, 0))
            d = self.dilated_convs[i](d)

            # condition
            cond = self.cond_convs[i](en)
            d = self._condition(d, cond)

            assert d.size(2) % 2 == 0, "Need to cut input in half"
            m = d.size(2) // 2
            d_sigmoid = torch.sigmoid(d[:, :, :m])
            d_tanh = torch.tanh(d[:, :, m:])
            d = d_sigmoid * d_tanh

            l += self.residual_convs[i](d)
            s += self.skip_convs[i](d)

        s = torch.relu(s)
        s = self.end_conv_1(s)
        s = self._condition(s, self.end_conv_2(en))
        s = torch.relu(s)

        return s

    def forward(self, input):
        x = self.wavenet(input)
        
        # l = 2 ** (self.layers - 1)
        out = x[:, :, -SR:]
        return out

    def _condition(self, x, encoding):
        """Condition the input on the encoding.
        Args:
        x: The [mb, length, channels] float tensor input.
        encoding: The [mb, encoding_length, channels] float tensor encoding.
        Returns:
        The output after broadcasting the encoding to x's shape and adding them.
        """
        mb, channels, length = x.size()
        enc_mb, enc_channels, enc_length = encoding.size()
        assert enc_mb == mb
        assert enc_channels == channels

        # TODO: maybe try transposing dim1,2 instead of just flipping?
        encoding = encoding.view([mb, channels, 1, enc_length])
        x = x.view([mb, channels, -1, enc_length])
        x += encoding
        x = x.view([mb, channels, length])
        return x

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.dtype = type
        super().cpu()
