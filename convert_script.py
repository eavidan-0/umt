from umt_model import *
from UmtDataset import *
import librosa
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *

from time import sleep
sleep(2)

dtype = torch.FloatTensor
ltype = torch.LongTensor

use_cuda = torch.cuda.is_available()
model = load_latest_model_from('snapshots', use_cuda=use_cuda)
model.train = False

print('model: ', model)
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
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=0,  # num_workers=8,
                                                 pin_memory=False)

        # generated = model.generate_fast(num_samples=16000,
        #                                 first_samples=data,
        #                                 progress_callback=prog_callback,
        #                                 progress_interval=1000,
        #                                 temperature=1.0,
        #                                 regularize=0.)

        generated = map(model.forward, iter(dataloader))
        generated = mu_law_expansion(list(generated), model.classes)

        out_path = GENERATION_OUTPUTS + filename + \
            '.' + DOMAINS[domain_index] + '.wav'
        print(out_path, generated)

        lr.output.write_wav(out_path, generated, sr=SR)


def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")
