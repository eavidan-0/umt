from umt_model import *
import librosa
from wavenet_model import *
import itertools

from audio_data import WavenetDataset
from umt_training import *

import torch

from time import sleep
sleep(2)

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

(model, __) = load_latest_model_from('snapshots', use_cuda=use_cuda)
model.train = False

classes = model.classes

print('parameter count: ', model.parameter_count())

if use_cuda:
    print("move model to gpu")
    model.cuda()
    model = torch.nn.parallel.DataParallel(
        model, device_ids=list(range(NUM_GPU)))

GENRATION_BASE = "./conversions/"
GENERATION_INPUTS = GENRATION_BASE + "in"
GENERATION_OUTPUTS = GENRATION_BASE + "out"

try:
    os.makedirs(GENERATION_OUTPUTS)
except OSError:
    pass

input_files = list_all_audio_files(GENERATION_INPUTS)


def data_to_type(data):
    domain_index, x, target = data

    x = Variable(x.type(dtype))
    target = Variable(target.type(ltype)).squeeze()
    domain_index = Variable(domain_index.type(ltype))

    return (domain_index, x, target)


for in_file in input_files:
    filename = os.path.splitext(os.path.basename(in_file))[0]
    print(filename)

    for domain_index in range(len(DOMAINS)):
        # Important: this is a wavenet dataset for a single domain
        dataset = WavenetDataset(dataset_file=GENRATION_BASE + filename + '.npz',
                                 item_length=SR,
                                 target_length=SR,
                                 file_location=in_file,
                                 train=False,
                                 domain_index=domain_index,
                                 test_stride=1)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 num_workers=4,  # num_workers=8,
                                                 pin_memory=False)

        i = 0
        total = len(dataset) // BATCH_SIZE
        total = 16 // BATCH_SIZE
        print (total, "samples")

        data = map(data_to_type, iter(dataloader))
        generated = map(model.forward, iter(data))
        generated = map(lambda x: convert_output_to_signal(
            x, classes).flatten(), generated)
        generated = list(itertools.islice(generated, total))
        generated = np.concatenate(generated)

        # convert data to signal...
        generated = decode_mu(generated, classes)

        out_path = GENERATION_OUTPUTS + "/" + filename + \
            '.' + DOMAINS[domain_index].replace(" ", "") + '.wav'
        print(out_path)

        lr.output.write_wav(out_path, generated, sr=SR)
