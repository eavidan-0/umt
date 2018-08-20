from wavenet_model import *
from audio_data import *
import numpy as np
import torch
import torch.nn.functional as F
import librosa as lr

# Umt downsampled by x12.5, but need divisability
# Nsynth used x32
DOWNSAMPLE_FACTOR = 40
ENC_LEN = SR / DOWNSAMPLE_FACTOR
if not ENC_LEN == int(ENC_LEN):
    raise ValueError("SR not divisable") 

ENC_LEN = int(ENC_LEN)

ENCODER_LAYERS = 8
ENCODER_OUTPUT_LENGTH = 2 ** (ENCODER_LAYERS - 1)
ENC_LEN = 64

DOWNSAMPLE_FACTOR = ENCODER_OUTPUT_LENGTH // ENC_LEN
if not DOWNSAMPLE_FACTOR == int(DOWNSAMPLE_FACTOR):
    raise ValueError("ENC_LEN not divisable") 

DOWNSAMPLE_FACTOR = int(DOWNSAMPLE_FACTOR)

class UmtModel(nn.Module):
    def __init__(self, dtype, classes=256, train=True):
        super(UmtModel, self).__init__()

        self.dtype = dtype
        self.classes = classes
        self.is_training = train

        self.encoder = WaveNetModel(blocks=3,
                                    classes=self.classes,
                                    layers=ENCODER_LAYERS,
                                    output_length=ENCODER_OUTPUT_LENGTH,
                                    dtype=dtype)

        decoders = [WaveNetModel(blocks=2,
                                 layers=5,
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

    def encode(self, input_tuple):
        torch.set_grad_enabled(self.is_training)

        domain_index_tensor, input, _ = input_tuple
        domain_index = domain_index_tensor.data[0]

        assert domain_index < len(
            self.decoders), "Unknown domain #%d" % domain_idx
        assert all([d == domain_index for d in domain_index_tensor.data]
                   ), "Mixed domain batch encountered"

        # Run through encoder
        enc = self.encoder.forward(input)
        return self.post_encode(enc)

    def get_encoder(self):
        return self.encoder, lambda enc: self.post_encode(enc)

    def post_encode(self, enc):
        latent = F.avg_pool1d(enc, kernel_size=DOWNSAMPLE_FACTOR)
        return enc

    def forward(self, input_tuple):
        domain_index_tensor, input, _ = input_tuple
        domain_index = domain_index_tensor.data[0]
        
        latent = self.encode(input_tuple)

        # Upsample back to original sampling rate
        # TODO: maybe everything SR? input_size[2]
        upsampled_latent = F.interpolate(latent, size=SR, mode='nearest')

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

    def cuda(self, device=None, type=torch.cuda.FloatTensor):
        self.encoder.cuda(device, type)
        for d in self.decoders:
            d.cuda(device, type)

        super().cuda(device)
