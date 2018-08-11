from umt_model import *
from audio_data import *
import torch
import numpy as np

DATASET_SOURCE_BASE = "/home/eavidan/Music/"
DATASET_LOCATION = "./datasets/"

BATCH_SIZE = 16

class UmtDataset(torch.utils.data.Dataset):
    def __init__(self,
                 item_length,
                 target_length,
                 batch_size=BATCH_SIZE,
                 dtype=np.uint8,
                 train=True,
                 test_stride=100):

        self._batch_size = batch_size
        self._datasets = []

        for domain_index in range(len(DOMAINS)):
            domain = DOMAINS[domain_index]
            data = WavenetDataset(dataset_file=DATASET_LOCATION + domain + '.npz',
                                  item_length=item_length,
                                  target_length=target_length,
                                  file_location=DATASET_SOURCE_BASE + domain,
                                  domain_index=domain_index,
                                  dtype=dtype,
                                  train=train,
                                  test_stride=test_stride)

            self._datasets.append(data)

        self._ds_num = len(DOMAINS)
        self._calculate_length()

    def _calculate_length(self):
        # HACK: limit to smallest DATASET of all
        self._ds_len = min([ds._length for ds in self._datasets])
        self._ds_len = int(self._ds_len / self._batch_size) * self._batch_size

        self._length = self._ds_len * self._ds_num

    def set_item_length(self, l):
        for ds in self._datasets:
            ds.set_item_length(l)
        self._calculate_length()

    def __getitem__(self, idx):
        batch_ind = int(idx / self._batch_size)

        ds_ind = batch_ind % self._ds_num
        ds_batch_ind = int(batch_ind / self._ds_num)

        ds_idx = ds_batch_ind * self._batch_size + (idx % self._batch_size)
        return self._datasets[ds_ind].__getitem__(ds_idx)

    def __len__(self):
        return self._length
