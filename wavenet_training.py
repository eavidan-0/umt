import torch
import torch.optim as optim
import torch.utils.data
import time
import os
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *
from umt_model import *

use_cuda = torch.cuda.is_available()


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


NUM_GPU = 4
CONFUSION_LOSS_WEIGHT = 10 ** -2


class DomainClassifier(nn.Module):
    def __init__(self, classes, bias=True):
        super(DomainClassifier, self).__init__()
        self.classes = classes

        self.conv_1 = nn.Conv1d(in_channels=classes,
                                out_channels=classes,
                                kernel_size=3,
                                bias=bias)

        self.conv_2 = nn.Conv1d(in_channels=classes,
                                out_channels=classes,
                                kernel_size=3,
                                bias=bias)
        
        self.conv_3 = nn.Conv1d(in_channels=classes,
                                out_channels=len(DOMAINS),
                                kernel_size=3,
                                bias=bias)

    def forward(self, latent):
        print(latent.size())
        x = latent

        x = self.conv_1(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_2(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_3(x)
        x = F.elu(x, alpha=1.0)

        print (x.size())
        x = F.avg_pool1d(x, kernel_size=ENC_LEN)
        print (x.size())
        raise OSError()
        return x


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=100,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model
        self.train_model = model if not use_cuda else nn.parallel.DataParallel(
            model, device_ids=list(range(NUM_GPU)))
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.domain_classifier = DomainClassifier(classes=model.classes)
        self.classifier_optimizer = self.optimizer_type(
            params=self.domain_classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model_optimizer = self.optimizer_type(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.logger = logger
        self.logger.trainer = self
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype

    def train(self,
              batch_size,
              epochs=100,
              continue_training_at_step=0):
        self.train_model.train()
        print("dataset length is", len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      #   shuffle=True,
                                                      num_workers=2,  # num_workers=8,
                                                      pin_memory=False)
        step = continue_training_at_step
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            for data in iter(self.dataloader):
                domain_index, x, target, one_hot_target, one_hot_domain_index = data

                x = Variable(x.type(self.dtype))
                # target = Variable(target.view(-1).type(self.ltype))
                target = Variable(target.type(self.ltype)).squeeze()
                one_hot_target = Variable(one_hot_target.type(self.ltype)).squeeze()
                one_hot_domain_index = Variable(one_hot_domain_index.type(self.ltype)).squeeze()

                # Pass through domain confusion model
                original_latent = self.train_model.encode(data)
                pred_domain = self.domain_classifier(original_latent)

                classifier_loss = F.cross_entropy(pred_domain, one_hot_domain_index)
                self.classifier_optimizer.zero_grad()
                classifier_loss.backward()
                self.classifier_optimizer.step()

                # Pass through network now
                output = self.train_model(data).squeeze()
                pred_domain = self.domain_classifier(
                    original_latent)  # same latent!

                classifier_loss = F.cross_entropy(pred_domain, one_hot_domain_index)
                model_loss = F.cross_entropy(output, one_hot_target)

                loss = model_loss - CONFUSION_LOSS_WEIGHT * classifier_loss
                self.model_optimizer.zero_grad()
                loss.backward()
                loss = loss.item()

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(
                        self.train_model.parameters(), self.clip)
                self.model_optimizer.step()
                step += 1

                # time step duration:
                if step % 10 == 0:
                    toc = time.time()
                    print("step", step, "loss: ", loss, "one training step does take approximately " +
                          str((toc - tic) * 0.1) + " seconds)")

                    tic = toc

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime(
                        "%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path +
                               '/' + self.snapshot_name + '_' + time_string)

                self.logger.log(step, loss)

    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        for (x, target) in iter(self.dataloader):
            x = Variable(x.type(self.dtype))
            target = Variable(target.view(-1).type(self.ltype))

            output = self.model(x)
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            total_loss += loss.data[0]

            predictions = torch.max(output, 1)[1].view(-1)
            correct_pred = torch.eq(target, predictions)
            accurate_classifications += torch.sum(correct_pred).data[0]
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        avg_loss = total_loss / len(self.dataloader)
        avg_accuracy = accurate_classifications / \
            (len(self.dataset) * self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy


def generate_audio(model,
                   length=8000,
                   temperatures=[0., 1.]):
    samples = []
    for temp in temperatures:
        samples.append(model.generate_fast(length, temperature=temp))
    samples = np.stack(samples, axis=0)
    return samples
