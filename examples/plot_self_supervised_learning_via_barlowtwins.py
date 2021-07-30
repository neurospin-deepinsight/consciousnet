# -*- coding: utf-8 -*-
"""
Barlow Twins: Self-Supervised Learning for clustering
=====================================================

Credit: A Grigis

A simple example on how to use the Barlow Twins to learn data representation
in an unsupervised way via redundancy reduction that in turns is used for
clustering using a simple linear layer to associate a learned representation
with a label.

The `test` variable must be set to False to run a full training.
"""

import os
import sys
import time
import json
import math
import types
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms
from consciousnet.augmentation import ContrastiveImageTransform
from consciousnet.optim import LARS
from consciousnet.models import BarlowTwins

test = True
n_epochs = 3 if test else 1000
batch_size = 10
learning_rate_weights = 0.2
learning_rate_biases = 0.0048
weight_decay = 1e-6
lambd = 0.0051
projector = "64-64"
print_freq = 10
datasetdir = "/tmp/minst"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# CIFAR10 dataset
# ---------------
#
# Fetch/load the CIFAR10 dataset.

def imshow(img):
    """ Unnormalize image and display it.
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataset = torchvision.datasets.CIFAR10(
    root=datasetdir, train=True, download=True,
    transform=ContrastiveImageTransform(50))
if test:
    dataset = torch.utils.data.random_split(
        dataset, [100, len(dataset) - 100])[0]
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=1,
    pin_memory=True)

dataiter = iter(dataloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images[0][:4]))

#############################################################################
# Training
# --------
#
# Create/train a simple conv network replacing the fully connected layer
# by a projection head.

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc = nn.Linear(1296, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def adjust_learning_rate(n_epochs, batch_size, learning_rate_weights,
                         learning_rate_biases, optimizer, loader, step):
    max_steps = n_epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]["lr"] = lr * learning_rate_weights
    optimizer.param_groups[1]["lr"] = lr * learning_rate_biases


def exclude_bias_and_norm(p):
    return p.ndim == 1


model = BarlowTwins(
    model=ConvNet(), fc_layer_name="fc", fc_in_features=1296,
    projector=projector, batch_size=batch_size, lambd=lambd).to(device)
print(model)
param_weights = []
param_biases = []
for param in model.parameters():
    if param.ndim == 1:
        param_biases.append(param)
    else:
        param_weights.append(param)
parameters = [{"params": param_weights}, {"params": param_biases}]
optimizer = LARS(parameters, lr=0, weight_decay=weight_decay,
                 weight_decay_filter=exclude_bias_and_norm,
                 lars_adaptation_filter=exclude_bias_and_norm)
start_time = time.time()
if device.type == "cpu":
    scaler = None
else:
    scaler = torch.cuda.amp.GradScaler()
for epoch in range(n_epochs):
    for step, ((y1, y2), _) in enumerate(
            dataloader, start=epoch * len(dataloader)):
        y1 = y1.to(device, non_blocking=True)
        y2 = y2.to(device, non_blocking=True)
        adjust_learning_rate(n_epochs, batch_size, learning_rate_weights,
                             learning_rate_biases, optimizer, dataloader,
                             step)
        optimizer.zero_grad()
        if device.type == "cpu":
            loss = model.forward(y1, y2)
        else:
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if step % print_freq == 0:
            stats = dict(epoch=epoch, step=step,
                         lr_weights=optimizer.param_groups[0]["lr"],
                         lr_biases=optimizer.param_groups[1]["lr"],
                         loss=loss.item(),
                         time=int(time.time() - start_time))
            print(json.dumps(stats))

#############################################################################
# Evaluation: linear classification
# ---------------------------------
#
# Train a linear probe on the representations learned by Barlow Twins. Freeze
# the weights of the resnet.

class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions for the specified
    values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


n_epochs = 3 if test else 100
lr_classifier = 0.3

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(50),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])
transform_val = transforms.Compose([
    transforms.CenterCrop(50),
    transforms.ToTensor(),
    normalize
])
ds_train = torchvision.datasets.CIFAR10(
    root=datasetdir, train=True, download=True, transform=transform_train)
ds_val = torchvision.datasets.CIFAR10(
    root=datasetdir, train=False, download=True, transform=transform_val)
if test:
    ds_train = torch.utils.data.random_split(
        ds_train, [100, len(ds_train) - 100])[0]
    ds_val = torch.utils.data.random_split(
        ds_val, [100, len(ds_val) - 100])[0]
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ["train", "val"]}

reference_state_dict = model.backbone.state_dict()
model = ConvNet().to(device)
print(model)
missing_keys, unexpected_keys = model.load_state_dict(
    reference_state_dict, strict=False)
assert (
    missing_keys == ["fc.weight", "fc.bias"] and
    unexpected_keys == [])
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()
model.requires_grad_(False)
model.fc.requires_grad_(True)
classifier_parameters, model_parameters = [], []
for name, param in model.named_parameters():
    if name in {"fc.weight", "fc.bias"}:
        classifier_parameters.append(param)
    else:
        model_parameters.append(param)
criterion = nn.CrossEntropyLoss().to(device)
param_groups = [dict(params=classifier_parameters, lr=lr_classifier)]
optimizer = optim.SGD(
    param_groups, 0, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
best_acc = argparse.Namespace(top1=0, top5=0)
start_time = time.time()
for epoch in range(n_epochs):
    model.eval()
    for step, (y, labels) in enumerate(
            dataloaders["train"], start=epoch * len(dataloaders["train"])):
        output = model(y.to(device, non_blocking=True))
        loss = criterion(output, labels.to(device, non_blocking=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % print_freq == 0:
            pg = optimizer.param_groups
            lr_classifier = pg[0]["lr"]
            lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
            stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                         lr_classifier=lr_classifier, loss=loss.item(),
                         time=int(time.time() - start_time))
            print(json.dumps(stats))

    # Evaluate
    model.eval()
    top1 = AverageMeter("Acc@1")
    top5 = AverageMeter("Acc@5")
    with torch.no_grad():
        for y, labels in dataloaders["val"]:
            output = model(y.to(device, non_blocking=True))
            acc1, acc5 = accuracy(
                output, labels.to(device, non_blocking=True), topk=(1, 5))
            top1.update(acc1[0].item(), y.size(0))
            top5.update(acc5[0].item(), y.size(0))
    best_acc.top1 = max(best_acc.top1, top1.avg)
    best_acc.top5 = max(best_acc.top5, top5.avg)
    stats = dict(
        epoch=epoch, acc1=top1.avg, acc5=top5.avg,
        best_acc1=best_acc.top1, best_acc5=best_acc.top5)
    print(json.dumps(stats))

    # Sanity check
    model_state_dict = model.state_dict()
    for k in reference_state_dict:
        assert torch.equal(
            model_state_dict[k].cpu(), reference_state_dict[k].cpu()), k

    scheduler.step()
    state = dict(
        epoch=epoch + 1, best_acc=best_acc, model=model.state_dict(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
