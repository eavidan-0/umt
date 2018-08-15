from wavenet_model import *
import numpy as np
import torch
import torch.nn.functional as F
import librosa as lr

DOMAINS = ["Ed Sheeran", "Metallica"]

ENC_LEN = 64
POOL_KERNEL = 50
SR = 16000


class UmtModel(nn.Module):
    def __init__(self, dtype, classes=256, train=True):
        super(UmtModel, self).__init__()

        self.classes = classes
        self.is_training = train

        # TODO: is this really the correct output?
        # Cause they said 12.5 downsample, not 250
        self.encoder = WaveNetModel(blocks=3,
                                    classes=self.classes,
                                    output_length=ENC_LEN * POOL_KERNEL,
                                    dtype=dtype)

        decoders = [WaveNetModel(blocks=4,
                                 classes=self.classes,
                                 output_length=SR,
                                 dtype=dtype) for _ in DOMAINS]
        self.decoders = nn.ModuleList(modules=decoders)

        d = self.decoders[0]
        e = self.encoder

        self.receptive_field = [e.receptive_field, d.receptive_field]
        self.output_length = [e.output_length, d.output_length]

        # self.item_length = model.receptive_field[0] + model.output_length[1] - 1
        self.item_length = SR
        self.target_length = SR

    def forward(self, input_tuple):
        torch.set_grad_enabled(self.is_training)

        domain_index_tensor, input, target = input_tuple
        domain_index = domain_index_tensor.data[0]
        input_size = input.size()

        assert domain_index < len(
            self.decoders), "Unknown domain #%d" % domain_idx
        assert all([d == domain_index for d in domain_index_tensor.data]
                   ), "Mixed domain batch encountered"

        # Run through encoder
        enc = self.encoder.forward(input)
        latent = F.avg_pool1d(enc, kernel_size=POOL_KERNEL)

        # Only if training: DOMAIN CLASSIFIER
        if self.is_training:
            # TODO: get it from outside so it can train separately
            pass

        # Upsample back to original sampling rate
        # TODO: maybe everything SR? input_size[2]
        upsampled_latent = F.interpolate(latent, size=SR, mode='nearest')

        # Run through domain decoder
        out = self.decoders[domain_index].forward(upsampled_latent)

        # TODO: mu-law again?

        # decode
        if not self.is_training:
            # TODO: or encode? or nothing at all?
            out = mu_law_expansion(out, self.classes)

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

    def cuda(self, device=None, type=torch.cuda.FloatTensor):
        self.encoder.cuda(device, type)
        for d in self.decoders:
            d.cuda(device, type)

        super().cuda(device)
