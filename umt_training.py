import torch
import torch.optim as optim
import torch.utils.data
import time
import os
from functools import reduce
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *
from umt_model import *
from random import random
import itertools
from domain_classifier import DomainClassifier

use_cuda = torch.cuda.is_available()
NUM_GPU = 4

if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(list(range(NUM_GPU)))[
        1:-1].replace(" ", "")

CONFUSION_LOSS_WEIGHT = 10 ** -2
INIT_LR = 10 ** -3
LR_DECAY = 0.98


class UmtTrainer:
    def __init__(self,
                 model,
                 datasets,
                 optimizer=optim.Adam,
                 lr=INIT_LR,
                 weight_decay=0,
                 gradient_clipping=5,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=100,
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model
        self.train_model = model
        self.datasets = datasets
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        self.encoder = model.encoder
        if use_cuda:
            self.train_model = nn.parallel.DataParallel(
                self.train_model, device_ids=list(range(NUM_GPU)))
            self.encoder = nn.parallel.DataParallel(
                self.encoder, device_ids=list(range(NUM_GPU)))

        self.model_optimizer = self.optimizer_type(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype

        self.decay_lr_times = 0
        self.create_domain_classifier()

    def train(self,
              batch_size,
              epochs=1000,
              start_epoch=0):
        self.train_model.train()
        dataloaders = list(
            map(lambda ds: create_dataloader(ds, batch_size), self.datasets))

        ds_length = min(map(lambda ds: len(ds), self.datasets))
        data_size = len(self.datasets) * (ds_length // batch_size * batch_size)

        print ("data length", data_size)
        self.snapshot_interval = data_size // batch_size
        step = int(start_epoch * self.snapshot_interval)

        # return to previous params
        for _ in range(step // 10000):
            self.decay_lr()

        snapshot_prefix = self.snapshot_path + '/' + self.snapshot_name + '_'

        for current_epoch in range(start_epoch, epochs):
            print("epoch", current_epoch)

            if current_epoch > start_epoch:
                time_string = time.strftime("%Y-%m-%d_%H-%M", time.gmtime())
                torch.save(self.model, snapshot_prefix + time_string)

            # New classifier - better confusion?
            # self.create_domain_classifier()

            # Shuffle entire batches to ensure same domain index
            for data in roundrobin(dataloaders, halt_on_first=True):
                domain_index, x, target = data

                x = Variable(x.type(self.dtype))
                # target = Variable(target.view(-1).type(self.ltype))
                target = Variable(target.type(self.ltype)).squeeze()
                domain_index = Variable(domain_index.type(self.ltype))

                data = (domain_index, x, target)

                # Pass through domain confusion model
                original_latent = self.encoder(x)
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

                # why did UMT subtract? adversarial?
                loss = model_loss - CONFUSION_LOSS_WEIGHT * classifier_loss
                self.model_optimizer.zero_grad()
                loss.backward()
                loss = loss.item()

                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(
                        self.train_model.parameters(), self.clip)
                self.model_optimizer.step()
                step += 1

                if step % 10000 == 0:
                    self.decay_lr()

                print("[EPOCH %d; STEP %d]: loss %.3f classifier_loss %.3f    [%r]" %
                      (current_epoch, step, loss, classifier_loss.item(), DOMAINS[domain_index[0]]))

    def decay_lr(self):
        self.decay_lr_times += 1

        new_lr = INIT_LR if self.decay_lr_times % 250 == 0 else self.lr * LR_DECAY

        self.lr = new_lr
        print ("DECAY LR", new_lr)

        self.set_lr(self.classifier_optimizer)
        self.set_lr(self.model_optimizer)

    def set_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr

    def create_domain_classifier(self):
        self.domain_classifier = DomainClassifier(classes=self.model.classes)
        if use_cuda:
            self.domain_classifier = self.domain_classifier.cuda()

        self.classifier_optimizer = self.optimizer_type(
            params=self.domain_classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def create_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,  # TODO: maybe batches?
                                       num_workers=4,
                                       drop_last=True,
                                       pin_memory=False)


def randomize(iterable):
    return sorted(iterable, key=lambda k: random())


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def roundrobin(iterables, halt_on_first=False):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle([iter(it).__next__ for it in iterables])
    while num_active:
        try:
            for n in nexts:
                yield n()
        except StopIteration:
            if halt_on_first:
                return

            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))
