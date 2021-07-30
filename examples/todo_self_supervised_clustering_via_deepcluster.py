# -*- coding: utf-8 -*-
"""
Self supervised clustering
==========================

Credit: A Grigis

In this example we illustrate the unsupervised training of convolutional
neural networks described in the paper Deep Clustering for Unsupervised
Learning of Visual Features.


The `test` variable must be set to False to run a full training.
"""

import os
import sys
import time
import copy
import collections
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision
from consciousnet.sampler import UniformLabelSampler
from consciousnet.models import DeepCluster

test = True
batch_size = 10
pca_size = 15
n_clusters = 10
adam_lr = 0.001
n_epochs = 3 if test else 30
avoid_empty_clusters = False
uniform_sampling = False
datasetdir = "/tmp/minst"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# CIFAR10 dataset
# ---------------
#
# Fetch/load the CIFAR10 dataset.

if uniform_sampling:
    sampler = UniformLabelSampler
else:
    sampler = "random"
ds_train = torchvision.datasets.CIFAR10(
    root=datasetdir, train=True, download=True)
ds_val = torchvision.datasets.CIFAR10(
    root=datasetdir, train=False, download=True)
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=batch_size, sampler=sampler, num_workers=1)
        for x in ["train", "val"]}

#############################################################################
# Training
# --------
#
# Create/train a simple supervised classification MLP model and the same
# unsupervised model with pseudo-label computed at each epoch using a K-means.

def train_model(dataloaders, model, device, criterion, optimizer,
                scheduler=None, n_epochs=100, checkpointdir=None,
                save_after_epochs=1, board=None, board_updates=None,
                load_best=False):
    """ General function to train a model and display training metrics.

    Parameters
    ----------
    dataloaders: dict of torch.utils.data.DataLoader
        the train & validation data loaders.
    model: nn.Module
        the model to be trained.
    device: torch.device
        the device to work on.
    criterion: torch.nn._Loss
        the criterion to be optimized.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    n_epochs: int, default 100
        the number of epochs.
    checkpointdir: str, default None
        a destination folder where intermediate models/histories will be
        saved.
    save_after_epochs: int, default 1
        determines when the model is saved and represents the number of
        epochs before saving.
    board: brainboard.Board, default None
        a board to display live results.
    board_updates: list of callable, default None
        update displayed item on the board.
    load_best: bool, default False
        optionally load the best model regarding the loss.
    """
    since = time.time()
    if board_updates is not None:
        board_updates = listify(board_updates)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    dataset_sizes = {x: len(dataloaders[x]) for x in ["train", "val"]}
    model = model.to(device)
    for epoch in range(n_epochs):
        print("Epoch {0}/{1}".format(epoch, n_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  
            else:
                model.eval()   
            running_loss = 0.0
            for batch_data, batch_labels in dataloaders[phase]:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward:
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs, layer_outputs = model(batch_data)
                    loss, extra_loss = criterion(outputs, batch_labels)
                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item() * batch_data[0].size(0)
            if scheduler is not None and phase == "train":
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            print("{0} Loss: {1:.4f}".format(phase, epoch_loss))
            if board is not None:
                board.update_plot("loss_{0}".format(phase), epoch, epoch_loss)
            # Display validation classification results
            if board_updates is not None and phase == "val":
                for update in board_updates:
                    update(model, board, outputs, layer_outputs)
            # Deep copy the best model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        # Save intermediate results
        if checkpointdir is not None and epoch % save_after_epochs == 0:
            outfile = os.path.join(
                checkpointdir, "model_{0}.pth".format(epoch))
            checkpoint(
                model=model, outfile=outfile, optimizer=optimizer,
                scheduler=scheduler, epoch=epoch, epoch_loss=epoch_loss)
        print()
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val loss: {:4f}".format(best_loss))
    # Load best model weights
    if load_best:
        model.load_state_dict(best_model_wts)


def listify(data):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    arr: list or array
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(data, list) or isinstance(data, tuple):
        return data
    else:
        return [data]


def checkpoint(model, outfile, optimizer=None, scheduler=None,
               **kwargs):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: nn.Module
        the model to be saved.
    outfile: str
        the destination file name.
    optimizer: torch.optim.Optimizer
        the optimizer.
    scheduler: torch.optim.lr_scheduler, default None
        the scheduler.
    kwargs: dict
        others parameters to be saved.
    """
    kwargs.update(model=model.state_dict())
    if optimizer is not None:
        kwargs.update(optimizer=optimizer.state_dict())
    if scheduler is not None:
        kwargs.update(scheduler=scheduler.state_dict())
    torch.save(kwargs, outfile)


class FKmeans(object):
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit(self, data):
        n_data, d = data.shape
        self.clus = faiss.Kmeans(d, self.n_clusters)
        self.clus.seed = np.random.randint(1234)
        self.clus.niter = 20
        self.clus.max_points_per_centroid = 10000000
        self.clus.train(data)

    def predict(self, data):
        _, I = self.clus.index.search(data, 1)
        losses = self.clus.obj
        print("k-means loss evolution: {0}".format(losses))
        return np.asarray([int(n[0]) for n in I])


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if avoid_empty_clusters:
    import faiss
    kmeans = FKmeans(n_clusters=n_clusters)
else:
    kmeans = KMeans(n_clusters=n_clusters, random_state=None, max_iter=20)
models = {}
models["self_supervised"] = DeepCluster(
    network=ConvNet(), clustering=kmeans,
    data_loader=dataloaders["train"], n_batchs=1, pca_dim=pca_size,
    assignment_logfile=None, device=device)
models["supervised"] = ConvNet()
for name, data in models.items():
    optimizer = optim.Adam(model.parameters(), lr=adam_lr)
    criterion = nn.CrossEntropyLoss()
    train_model(dataloaders, data["model"], device, criterion,
                optimizer, scheduler=None, n_epochs=n_epochs,
                checkpointdir=None, board=None, load_best=False)

#############################################################################
# Results
# -------
#
# Let us look at how the network performs on the whole validation dataset.

def test_model(dataloaders, model, device):
    """ General function to test a model.

    Parameters
    ----------
    dataloaders: dict of torch.utils.data.DataLoader
        the train & validation data loaders.
    model: nn.Module
        the trained model.
    device: torch.device
        the device to work on.
    """
    was_training = model.training
    model.eval()
    total, correct = (0, 0)
    labels, true_labels = [], []
    with torch.no_grad():
        for idx, (batch_data, batch_labels) in enumerate(dataloaders["val"]):
            batch_data = batch_data.to(device)
            true_labels.extend(batch_labels.numpy().tolist())
            batch_labels = batch_labels.to(device)
            outputs, layer_outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, dim=1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            labels.extend(batch_labels)
    print("Accuracy of the network on the validation images: %d %%" % (
        100 * correct / total))
    model.train(mode=was_training)
    labels = np.asarray(labels)
    true_labels = np.asarray(true_labels)
    return labels, true_labels


def get_members(y, n_clusters):
    """ Returns a dict with cluster numbers as keys and member entities
    as sorted numpy arrays.
    """
    labels = sorted(range(n_clusters))
    members = collections.OrderedDict()
    for lab in labels:
        members[lab] = (y == lab).astype(int)
    return members

def confusion_matrix(members1, members2):
    nb_elems_in_members1 = len(members1)
    nb_elems_in_members2 = len(members2)
    cmatrix = np.zeros((nb_elems_in_members1, nb_elems_in_members2))
    for idx1, vec1 in enumerate(members1.values()):
        for idx2, vec2 in enumerate(members2.values()):
            cmatrix[idx1, idx2] = dice_coef(vec1, vec2)
    return cmatrix


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return ((2. * intersection + smooth) /
            (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


for name, data in models.items():
    labels, true_labels = test_model(dataloaders, data["model"], device)
    if name == "self_supervised":
        y_true_members = get_members(true_labels, n_clusters)
        y_pred_members = get_members(labels, n_clusters)
        cmatrix = confusion_matrix(y_true_members, y_pred_members)
        np.set_printoptions(precision=2, suppress=True)
        print("Confusion matrix:", cmatrix)
        mapping = np.argmax(cmatrix, axis=1)
        overlap = np.max(cmatrix, axis=1)
        print("Mapping:", mapping)
        print("Labels:", sorted(np.unique(mapping)))
        print("Overlap:", overlap)
    else:
        print(classification_report(true_labels, labels))

plt.show()
