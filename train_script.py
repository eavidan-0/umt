from umt_model import *
from UmtDataset import *
from audio_data import *
from umt_training import *
from model_logging import *
from scipy.io import wavfile

from time import sleep
sleep(2)

DATASET_SOURCE_BASE = "/home/eavidan/Music/"
DATASET_LOCATION = "./datasets/"

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

    NUM_GPU = 4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(list(range(NUM_GPU)))[
        1:-1].replace(" ", "")

model = UmtModel(dtype)
print('parameter count: ', model.parameter_count())

# reload snapshot
start_epoch = 0
# model = load_latest_model_from('snapshots', use_cuda=use_cuda)

if use_cuda:
    print("move model to gpu")
    model.cuda()

datasets = []

for domain_index in range(len(DOMAINS)):
    domain = DOMAINS[domain_index]
    data = WavenetDataset(dataset_file=DATASET_LOCATION + domain + '.npz',
                          item_length=SR,
                          target_length=SR,
                          file_location=DATASET_SOURCE_BASE + domain,
                          domain_index=domain_index,
                          dtype=dtype,
                          train=True,
                          test_stride=5000000)

    datasets.append(data)

trainer = UmtTrainer(model=model,
                     datasets=datasets,
                     snapshot_path='snapshots',
                     snapshot_name='umt',
                     dtype=dtype,
                     ltype=ltype)

print('start training...')
trainer.train(batch_size=BATCH_SIZE,
              start_epoch=start_epoch)
