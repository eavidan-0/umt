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
from random import random
import itertools

use_cuda = torch.cuda.is_available()


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])


NUM_GPU = 4
CONFUSION_LOSS_WEIGHT = 0.1  # they did 0.01

INIT_LR = 10 ** -3
LR_DECAY = 0.98


class DomainClassifier(nn.Module):
    def __init__(self, classes, bias=True):
        super(DomainClassifier, self).__init__()

        self.classes = classes
        channels = classes // 8

        self.conv_1 = nn.Conv1d(in_channels=classes,
                                out_channels=channels,
                                kernel_size=3,
                                bias=bias)

        self.conv_2 = nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=3,
                                bias=bias)

        self.conv_3 = nn.Conv1d(in_channels=channels,
                                out_channels=len(DOMAINS),
                                kernel_size=3,
                                bias=bias)

    def forward(self, latent):
        x = latent

        x = self.conv_1(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_2(x)
        x = F.elu(x, alpha=1.0)
        x = self.conv_3(x)
        x = F.elu(x, alpha=1.0)

        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        x = x.squeeze()  # [batch, D]
        x = F.normalize(x, dim=1)

        return x


class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=INIT_LR,
                 weight_decay=0,
                 gradient_clipping=None,
                 logger=Logger(),
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=100,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model
        self.train_model = model
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.domain_classifier = DomainClassifier(classes=model.classes)
        self.encoder, self.post_encode = model.get_encoder()
        if use_cuda:
            self.train_model = nn.parallel.DataParallel(
                self.train_model, device_ids=list(range(NUM_GPU)))
            self.domain_classifier = self.domain_classifier.cuda()
            self.encoder = nn.parallel.DataParallel(
                self.encoder, device_ids=list(range(NUM_GPU)))

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
              epochs=1000,
              start_epoch=0):
        self.train_model.train()
        print("dataset length is", len(self.dataset))
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      batch_sampler=MultiDomainRandomSampler(
                                                          self.dataset, batch_size),
                                                      num_workers=2,  # num_workers=8,
                                                      pin_memory=False)
        self.snapshot_interval = len(self.dataset) / batch_size
        step = start_epoch * self.snapshot_interval

        # return to previous params
        for _ in range(start_epoch):
            self.decay_lr()

        for current_epoch in range(start_epoch, epochs):
            print("epoch", current_epoch)
            if current_epoch != 0:
                self.decay_lr()

            tic = time.time()
            # TODO: Shuffle entire batches to ensure same domain index
            for data in iter(self.dataloader):
                domain_index, x, target = data

                x = Variable(x.type(self.dtype))
                # target = Variable(target.view(-1).type(self.ltype))
                target = Variable(target.type(self.ltype)).squeeze()
                domain_index = Variable(domain_index.type(self.ltype))

                data = (domain_index, x, target)

                # Pass through domain confusion model
                original_latent = self.post_encode(self.encoder(x))
                pred_domain = self.domain_classifier(original_latent)

                classifier_loss = F.cross_entropy(pred_domain, domain_index)
                self.classifier_optimizer.zero_grad()
                classifier_loss.backward(retain_graph=True)
                self.classifier_optimizer.step()

                # Pass through network now (and updated classifier)
                pred_domain = self.domain_classifier(
                    original_latent)  # same enc!
                classifier_loss = F.cross_entropy(pred_domain, domain_index)

                output = self.train_model(data).squeeze()
                model_loss = F.cross_entropy(output, target)

                # why did UMT subtract?
                loss = model_loss + CONFUSION_LOSS_WEIGHT * classifier_loss
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
                    print("step", step, "loss: ", loss,
                          "conf_loss", classifier_loss.item())

                    tic = toc

                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string = time.strftime(
                        "%Y-%m-%d_%H-%M", time.gmtime())
                    torch.save(self.model, self.snapshot_path +
                               '/' + self.snapshot_name + '_' + time_string)

                self.logger.log(step, loss)

    def decay_lr(self):
        self.lr = self.lr * LR_DECAY
        print ("DECAY LR")

        self.set_lr(self.classifier_optimizer)
        self.set_lr(self.model_optimizer)

    def set_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= self.lr

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


class MultiDomainRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.length = len(data_source)
        self.batch_size = batch_size

        self.range_base = range(self.length / len(DOMAINS))

    def __iter__(self):
        D = len(DOMAINS)
        domain_batches = itertools.chain(map(lambda idx: self._get_domain_range(idx), range(D)))

        # provides randomness between domains, in batches
        return randomize(domain_ranges)

    def _get_domain_range(self, domain_idx):
        # provides randomness - inside the domain
        d_range = map(lambda idx: self._get_item_index(
            domain_idx, idx), self.range_base)

        rand_range = randomize(d_range)
        b_range = grouper(self.batch_size, rand_range)

        return b_range

    def _get_item_index(self, domain_idx, idx):
        D = len(DOMAINS)
        idx_in_batch = idx % self._batch_size
        batch_in_domain = int(idx / self.batch_size)
        batch_start = (D * batch_in_domain + domain_index) * self.batch_size

        return batch_start + idx_in_batch

    def __len__(self):
        return self.length
    
def randomize(iterable):
    return sorted(iterable, key=lambda k: random())

def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk