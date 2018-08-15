from umt_model import *
from UmtDataset import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from scipy.io import wavfile

from time import sleep
sleep(2)

print(1)
dtype = torch.FloatTensor
ltype = torch.LongTensor
print(2)

use_cuda = torch.cuda.is_available()
print(3, use_cuda)

if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
print(4)

model = UmtModel(dtype)
print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

print ('item_length', model.item_length)
print ('target_length', model.target_length)

# reload snapshot
continue_training_at_step = 0
# model = load_latest_model_from('snapshots', use_cuda=use_cuda)

if use_cuda:
    print("move model to gpu")
    model.cuda()

data = UmtDataset(item_length=model.item_length,
                  target_length=model.target_length,
                  train=True,
                  test_stride=500)

def generate_and_log_samples(step):
    sample_length = 32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")


logger = TensorboardLogger(log_interval=200,
                           validation_interval=400,
                           generate_interval=800,
                           generate_function=generate_and_log_samples,
                           log_dir="logs/chaconne_model")

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='umt_model',
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=BATCH_SIZE,
              epochs=10,
              continue_training_at_step=continue_training_at_step)
