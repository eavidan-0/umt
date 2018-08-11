from wavenet_model import *
import numpy as np
import torch
import torch.nn.functional as F

DOMAINS = ["Ed Sheeran", "Metallica"]
ENC_LEN = 64
POOL_KERNEL = 800
SR = 16000


class UmtModel(nn.Module):
    def __init__(self, dtype):
        super(UmtModel, self).__init__()

        # TODO: is this really the correct output?
        # Cause they said 12.5 downsample, not 250
        self.encoder = WaveNetModel(blocks=3,
                                    output_length=ENC_LEN * POOL_KERNEL,
                                    dtype=dtype)

        self.decoders = []
        for _ in range(len(DOMAINS)):
            self.decoders.append(WaveNetModel(blocks=4,
                                              output_length=SR,
                                              dtype=dtype))

        d = self.decoders[0]
        self.receptive_field = [
            self.encoder.receptive_field, d.receptive_field]
        self.output_length = [self.encoder.output_length, d.output_length]

    def forward(self, input):
        domain_index_tensor, input = input
        domain_index = domain_index_tensor.data[0]
        
        assert domain_index < len(
            self.decoders), "Unknown domain #%d" % domain_idx
        assert all([d == domain_index for d in domain_index_tensor.data]
                   ), "Mixed domain batch encountered"

        print ("remember you need to understand input size...", input.size())

        # TODO: Pitch modulation, only if training

        # Run through encoder
        enc = self.encoder.forward(input)
        latent = F.avg_pool1d(enc, kernel_size=POOL_KERNEL)

        # TODO: DOMAIN CLASSIFIER, from outside... only if training

        # Upsample back to original sampling rate
        upsampled_latent = F.upsample(latent, size=input_size, mode='nearest')

        # Run through domain decoder
        out = self.decoders[domain_index].forward(upsampled_latent)

        # TODO: mu-law again?
        return out

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def cpu(self, type=torch.FloatTensor):
        self.encoder.cpu(type)
        for d in self.decoders:
            d.cpu(type)

        super().cpu()
