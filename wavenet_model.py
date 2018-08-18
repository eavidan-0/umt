import os
import os.path
import time
from wavenet_modules import *
from audio_data import *
import math
import numpy as np


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
                 blocks,
                 classes,
                 output_length,
                 layers=6,
                 channels=128,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=True):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.channels = channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dtype = dtype

        # build model
        receptive_field = 1
        prev_dilation = 1

        self.dilated_convs = nn.ModuleList()
        # self.dilated_queues = []
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

                # dilated queues for fast generation
                # self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * dilation + 1,
                #                                         num_channels=channels,
                #                                         dilation=dilation,
                #                                         dtype=dtype))

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

        # self.output_length = 2 ** (layers - 1)
        self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input):

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

        # TODO: we need the next three lines? not in article..
        # x = F.relu(x)
        # x = self.end_conv_1(x)
        # x = F.relu(x)
        x = self.end_conv(x)

        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input)

        # reshape output
        [n, c, l] = x.size()
        l = self.output_length
        x = x[:, :, -l:]
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(n * l, c)

        return x

    def generate(self,
                 num_samples,
                 first_samples=None,
                 temperature=1.):
        self.eval()
        if first_samples is None:
            first_samples = self.dtype(1).zero_()
        generated = Variable(first_samples, volatile=True)

        num_pad = self.receptive_field - generated.size(0)
        if num_pad > 0:
            generated = constant_pad_1d(generated, self.scope, pad_start=True)
            print("pad zero")

        for i in range(num_samples):
            input = Variable(torch.FloatTensor(
                1, self.classes, self.receptive_field).zero_())
            input = input.scatter_(
                1, generated[-self.receptive_field:].view(1, -1, self.receptive_field), 1.)

            x = self.wavenet(input)[:, :, -1].squeeze()

            if temperature > 0:
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = Variable(torch.LongTensor([x]))  # np.array([x])
            else:
                x = torch.max(x, 0)[1].float()

            generated = torch.cat((generated, x), 0)

        generated = (generated / self.classes) * 2. - 1
        mu_gen = mu_law_expansion(generated, self.classes)

        self.train()
        return mu_gen

    def generate_fast(self,
                      num_samples,
                      first_samples=None,
                      temperature=1.,
                      regularize=0.,
                      progress_callback=None,
                      progress_interval=100):
        self.eval()
        if first_samples is None:
            first_samples = torch.LongTensor(1).zero_() + (self.classes // 2)
        first_samples = Variable(first_samples)

        # reset queues
        for queue in self.dilated_queues:
            queue.reset()

        num_given_samples = first_samples.size(0)
        total_samples = num_given_samples + num_samples

        input = Variable(torch.FloatTensor(1, self.classes, 1).zero_())
        input = input.scatter_(1, first_samples[0:1].view(1, -1, 1), 1.)

        # fill queues with given samples
        for i in range(num_given_samples - 1):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate)
            input.zero_()
            input = input.scatter_(
                1, first_samples[i + 1:i + 2].view(1, -1, 1), 1.).view(1, self.classes, 1)

            # progress feedback
            if i % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i, total_samples)

        # generate new samples
        generated = np.array([])
        regularizer = torch.pow(
            Variable(torch.arange(self.classes)) - self.classes / 2., 2)
        regularizer = regularizer.squeeze() * regularize
        tic = time.time()
        for i in range(num_samples):
            x = self.wavenet(input,
                             dilation_func=self.queue_dilate).squeeze()

            x -= regularizer

            if temperature > 0:
                # sample from softmax distribution
                x /= temperature
                prob = F.softmax(x, dim=0)
                prob = prob.cpu()
                np_prob = prob.data.numpy()
                x = np.random.choice(self.classes, p=np_prob)
                x = np.array([x])
            else:
                # convert to sample value
                x = torch.max(x, 0)[1][0]
                x = x.cpu()
                x = x.data.numpy()

            o = (x / self.classes) * 2. - 1
            generated = np.append(generated, o)

            # set new input
            x = Variable(torch.from_numpy(x).type(torch.LongTensor))
            input.zero_()
            input = input.scatter_(1, x.view(1, -1, 1),
                                   1.).view(1, self.classes, 1)

            if (i + 1) == 100:
                toc = time.time()
                print("one generating step does take approximately " +
                      str((toc - tic) * 0.01) + " seconds)")

            # progress feedback
            if (i + num_given_samples) % progress_interval == 0:
                if progress_callback is not None:
                    progress_callback(i + num_given_samples, total_samples)

        self.train()
        mu_gen = mu_law_expansion(generated, self.classes)
        return mu_gen

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
