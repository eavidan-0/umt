from wavenet_model import *
from encoder_model import *
from audio_data import *
import numpy as np
import torch
import torch.nn.functional as F
import librosa as lr


class UmtModel(nn.Module):
    def __init__(self, dtype, classes=256, train=True):
        super(UmtModel, self).__init__()

        self.dtype = dtype
        self.classes = classes
        self.is_training = train

        self.encoder = EncoderModel(classes=self.classes,
                                    dtype=dtype,
                                    bias=True)

        decoders = [WaveNetModel(layers=10,
                                 blocks=4,
                                 classes=self.classes,
                                 output_length=SR,
                                 dtype=dtype,
                                 bias=False) for _ in DOMAINS]
        self.decoders = nn.ModuleList(modules=decoders)

    def encode(self, input_tuple):
        torch.set_grad_enabled(self.is_training)

        domain_index_tensor, input, _ = input_tuple
        domain_index = domain_index_tensor.data[0]

        assert domain_index < len(
            self.decoders), "Unknown domain #%d" % domain_idx
        assert all([d == domain_index for d in domain_index_tensor.data]
                   ), "Mixed domain batch encountered"

        # Run through encoder
        enc = self.encoder(input)
        return enc

    def forward(self, input_tuple):
        domain_index_tensor, input, _ = input_tuple
        domain_index = domain_index_tensor.data[0]

        latent = self.encode(input_tuple)

        # Upsample back to original sampling rate
        upsampled_latent = F.interpolate(latent, size=SR, mode='nearest')

        # Run through domain decoder
        # Mu input and output
        upsampled_latent = quantize_data(upsampled_latent, self.classes)
        out = self.decoders[domain_index].forward(upsampled_latent)
        out = decode_mu(out, self.classes)  # TODO: or just expansion?

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
            d.cuda(device)

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
