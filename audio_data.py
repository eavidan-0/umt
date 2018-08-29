import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import librosa as lr
import bisect
import torch.nn as nn
import torch.nn.functional as F

from random import random, randint

DOMAINS = ["_flute", "_guitar", "_organ", "_vocal"]
DOMAIN_IDS = list(range(len(DOMAINS)))
SR = 16000
BATCH_SIZE = 16

class WavenetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_file,
                 item_length,
                 target_length,
                 domain_index,
                 sampling_rate=SR,
                 file_location=None,
                 classes=256,
                 mono=True,
                 normalize=False,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100):

        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

        self.dataset_file = dataset_file
        self._item_length = item_length
        self._test_stride = test_stride
        self.target_length = target_length
        self.classes = classes
        self.domain_index = domain_index
        self.sampling_rate = sampling_rate

        if not os.path.isfile(dataset_file):
            assert file_location is not None, "no location for dataset files specified"
            self.mono = mono
            self.normalize = normalize

            self.dtype = dtype
            self.create_dataset(file_location, dataset_file)
        else:
            # Unknown parameters of the stored dataset
            # TODO Can these parameters be stored, too?
            self.mono = None
            self.normalize = None

            self.dtype = None

        self.data = np.load(self.dataset_file, mmap_mode='r')
        self.start_samples = [0]
        self._length = 0
        self.calculate_length()
        self.train = train
        # assign every *test_stride*th item to the test set

    def create_dataset(self, location, out_file):
        print("create dataset from audio files at", location)
        self.dataset_file = out_file
        files = list_all_audio_files(location)
        processed_files = []
        for i, file in enumerate(files):
            print("  processed " + str(i) + " of " + str(len(files)) + " files")
            file_data, _ = lr.load(path=file,
                                   sr=self.sampling_rate,
                                   mono=self.mono)
            if self.normalize:
                file_data = lr.util.normalize(file_data)
            quantized_data = quantize_data(
                file_data, self.classes).astype(self.dtype)
            processed_files.append(quantized_data)

        np.savez(self.dataset_file, *processed_files)

    def calculate_length(self):
        start_samples = [0]
        for i in range(len(self.data.keys())):
            start_samples.append(
                start_samples[-1] + len(self.data['arr_' + str(i)]))
        available_length = start_samples[-1] - \
            (self._item_length - (self.target_length - 1)) - 1
        self._length = math.floor(available_length / self.target_length)
        self.start_samples = start_samples

    def set_item_length(self, l):
        self._item_length = l
        self.calculate_length()

    def __getitem__(self, idx):
        if self._test_stride < 2:
            sample_index = idx * self.target_length
        elif self.train:
            sample_index = idx * self.target_length + \
                math.floor(idx / (self._test_stride - 1))
        else:
            sample_index = self._test_stride * (idx + 1) - 1

        file_index = bisect.bisect_left(self.start_samples, sample_index) - 1
        if file_index < 0:
            file_index = 0
        if file_index + 1 >= len(self.start_samples):
            print("error: sample index " + str(sample_index) +
                  " is to high. Results in file_index " + str(file_index))
        position_in_file = sample_index - self.start_samples[file_index]
        end_position_in_next_file = sample_index + \
            self._item_length + 1 - self.start_samples[file_index + 1]

        if end_position_in_next_file < 0:
            file_name = 'arr_' + str(file_index)
            this_file = np.load(self.dataset_file, mmap_mode='r')[file_name]
            sample = this_file[position_in_file:position_in_file +
                               self._item_length + 1]
        else:
            # load from two files
            file1 = np.load(self.dataset_file, mmap_mode='r')[
                'arr_' + str(file_index)]
            file2 = np.load(self.dataset_file, mmap_mode='r')[
                'arr_' + str(file_index + 1)]
            sample1 = file1[position_in_file:]
            sample2 = file2[:end_position_in_next_file]
            sample = np.concatenate((sample1, sample2))

        # Only if training: Pitch modulation
        if self.train:
            # Reverse quantization and encoding
            print (sample)
            y = decode_mu(sample, self.classes)

            seg_length = int(self.sampling_rate *
                             (random() * 0.25 + 0.25))  # 1/4 -> 1/2
            s = randint(0, self.sampling_rate - seg_length)
            e = s + seg_length

            n_steps = random() - 0.5  # +- 0.5
            shifted = lr.effects.pitch_shift(
                y[s:e], sr=self.sampling_rate, n_steps=n_steps)
            y[s:e] = np.clip(shifted, -1, 1)

            sample = quantize_data(y, self.classes, mu=True)

        example = torch.from_numpy(sample).type(torch.LongTensor)
        input = example[:self._item_length].unsqueeze(0)
        target = example[-self.target_length:].unsqueeze(0)

        one_hot = torch.FloatTensor(self.classes, self._item_length).zero_()
        one_hot.scatter_(0, input, 1.)

        return self.domain_index, one_hot, target

    def __len__(self):
        test_length = math.floor(self._length / self._test_stride)
        if self.train:
            return self._length - test_length
        else:
            return test_length


def quantize_data(data, classes, mu=True):
    x = data
    if mu:
        x = mu_law_encoding(x, classes)

    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(x, bins) - 1
    return quantized

def decode_mu(data, classes):
    y = (data / classes) * 2. - 1
    y = mu_law_expansion(y, classes)
    return y;

def list_all_audio_files(location):
    audio_files = []
    if is_music_file(location):
        audio_files.append(location)
    else:
        for dirpath, dirnames, filenames in os.walk(location):
            for filename in [f for f in filenames if is_music_file(f)]:
                audio_files.append(os.path.join(dirpath, filename))

    if len(audio_files) == 0:
        print("found no audio files in " + location)
    return audio_files


def is_music_file(f):
    return f.endswith((".m4a", ".mp3", ".wav", ".aif", "aiff"))


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def convert_output_to_signal(x, classes):
    x = x.squeeze()
    dim = x.dim()
    x = x.transpose(dim - 2, dim - 1)

    prob = F.softmax(x, dim=dim - 1)  # map seconds to buckets
    prob = prob.cpu()
    np_prob = prob.data.numpy()    # Compute SM bucket for second

    x = np.apply_along_axis(
        lambda p: np.random.choice(classes, p=p), dim - 1, np_prob)
    return x

class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

