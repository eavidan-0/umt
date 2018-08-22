from wavenet_model import *
from encoder_model import *
from audio_data import *
import numpy as np
import torch
import torch.nn.functional as F
import librosa as lr

# Tasks
# [ ] Max Network size
# [ ] Encoder output + pooling
# [ ] Decoder output + mU
# [ ] Generation?


class UmtModel(nn.Module):
    def __init__(self, dtype, classes=256, train=True):
        super(UmtModel, self).__init__()

        self.dtype = dtype
        self.classes = classes
        self.is_training = train

        self.encoder = EncoderModel(blocks=3,
                                    layers=10,
                                    classes=self.classes,
                                    dtype=dtype,
                                    bias=False)

        decoders = [WaveNetModel(blocks=3,
                                 layers=10,
                                 output_length=SR,
                                 dilation_channels=32,
                                 residual_channels=16,
                                 skip_channels=16,
                                 classes=self.classes,
                                 dtype=dtype,
                                 bias=False) for _ in DOMAINS]
        self.decoders = nn.ModuleList(modules=decoders)

        self.receptive_field = self.decoders[0].receptive_field
        print ('receptive field', self.receptive_field)

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

        # Run through domain decoder
        # TODO: mu here or in dataset?
        out = self.decoders[domain_index].forward(latent)

        # TODO: ahem? mu? sampling rate? what what?
        out = F.interpolate(out, size=SR, mode='nearest')
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
