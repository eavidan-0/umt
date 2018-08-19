from umt_model import *
from UmtDataset import *
import librosa
from wavenet_model import *
import itertools

from audio_data import WavenetDataset
from wavenet_training import *

from time import sleep
sleep(2)

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

model = load_latest_model_from('snapshots', use_cuda=use_cuda)
model.train = False

if use_cuda:
    print("move model to gpu")
    model.cuda()

print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

GENRATION_BASE = "./conversions/"
GENERATION_INPUTS = GENRATION_BASE + "in"
GENERATION_OUTPUTS = GENRATION_BASE + "out"

try:
    os.makedirs(GENERATION_OUTPUTS)
except OSError:
    pass

input_files = list_all_audio_files(GENERATION_INPUTS)

for in_file in input_files:
    filename = os.path.splitext(os.path.basename(in_file))[0]
    print(filename)

    for domain_index in range(len(DOMAINS)):
        # Important: this is a wavenet dataset for a single domain
        dataset = WavenetDataset(dataset_file=GENRATION_BASE + filename + '.npz',
                                 item_length=model.item_length,
                                 target_length=model.target_length,
                                 file_location=in_file,
                                 train=False,
                                 domain_index=domain_index,
                                 test_stride=1)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=16,
                                                 shuffle=False,
                                                 num_workers=4,  # num_workers=8,
                                                 pin_memory=False)

        i = 0
        total = len(dataset)
        total = 16
        print (total, "samples")

        def prog_callback(x):
            i += 1
            print(str(100.0 * i / total) + "% generated")
            return x

        generated = map(model.forward, iter(dataloader))
        # generated = map(prog_callback, generated)
        generated = map(lambda x: convert_output_to_signal(
            x, model.classes), generated)
        generated = list(itertools.islice(generated, total))
        generated = np.concatenate(generated[:])

        # convert data to signal...
        generated = (generated / model.classes) * 2. - 1
        generated = mu_law_expansion(generated, model.classes)

        out_path = GENERATION_OUTPUTS + "/" + filename + \
            '.' + DOMAINS[domain_index].replace(" ", "") + '.wav'
        print(out_path)

        lr.output.write_wav(out_path, generated, sr=SR)
