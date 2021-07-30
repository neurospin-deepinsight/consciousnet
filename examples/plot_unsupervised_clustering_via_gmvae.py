# -*- coding: utf-8 -*-
"""
Unsupervised clustering with GMVAE
==================================

Credit: A Grigis

Unsupervised Gaussian Mixture Variational Auto-encoder (GMVAE) on a synthetic
dataset. In this example we attempt to replicate the work described in this
[blog](http://ruishu.io/2016/12/25/gmvae) inspired from this
[paper](https://arxiv.org/abs/1611.02648).

The `test` variable must be set to False to run a full training.
"""

import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from consciousnet.models import GMVAE
from consciousnet.losses import GMVAELoss

test = True
n_samples = 100
n_classes = 3
n_feats = 4
true_lat_dims = 2
fit_lat_dims = 5
snr = 10
batch_size = 10
adam_lr = 2e-3
n_epochs = 3 if test else 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# Synthetic dataset
# -----------------
#
# A Gaussian Linear multi-class synthetic dataset is generated as
# follows. The number of the latent dimensions used to generate the data can be
# controlled.

class GeneratorUniform(nn.Module):
    """ Generate multiple sources (channels) of data through a linear
    generative model:

    z ~ N(mu,sigma)
    for c_idx in n_channels:
        x_ch = W_ch(c_idx)
    where 'W_ch' is an arbitrary linear mapping z -> x_ch
    """
    def __init__(self, lat_dim=2, n_channels=2, n_feats=5, seed=100):
        super(GeneratorUniform, self).__init__()
        self.lat_dim = lat_dim
        self.n_channels = n_channels
        self.n_feats = n_feats
        self.seed = seed
        np.random.seed(self.seed)
        W = []
        for c_idx in range(n_channels):
            w_ = np.random.uniform(-1, 1, (self.n_feats, lat_dim))
            u, s, vt = np.linalg.svd(w_, full_matrices=False)
            w = (u if self.n_feats >= lat_dim else vt)
            W.append(torch.nn.Linear(lat_dim, self.n_feats, bias=False))
            W[c_idx].weight.data = torch.FloatTensor(w)
        self.W = torch.nn.ModuleList(W)

    def forward(self, z):
        if isinstance(z, list):
            return [self.forward(_) for _ in z]
        if type(z) == np.ndarray:
            z = torch.FloatTensor(z)
        assert z.size(dim=1) == self.lat_dim
        obs = []
        for c_idx in range(self.n_channels):
            x = self.W[c_idx](z)
            obs.append(x.detach())
        return obs


class SyntheticDataset(Dataset):
    def __init__(self, n_samples=500, lat_dim=2, n_feats=5, n_classes=2,
                 generatorclass=GeneratorUniform, snr=1, train=True):
        super(SyntheticDataset, self).__init__()
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.n_feats = n_feats
        self.n_classes = n_classes
        self.snr = snr
        self.train = train
        self.labels = []
        self.z = []
        self.x = []
        seed = 7 if self.train else 14
        np.random.seed(seed)
        locs = np.random.uniform(-5, 5, (self.n_classes, ))
        np.random.seed(seed)
        scales = np.random.uniform(0, 2, (self.n_classes, ))
        np.random.seed(seed)
        for k_idx in range(self.n_classes):
            self.z.append(
                np.random.normal(loc=locs[k_idx], scale=scales[k_idx],
                                 size=(self.n_samples, self.lat_dim)))
            self.generator = generatorclass(
                lat_dim=self.lat_dim, n_channels=1, n_feats=self.n_feats)
            self.x.append(self.generator(self.z[-1])[0])
            self.labels += [k_idx] * self.n_samples
        self.data = np.concatenate(self.x, axis=0)
        self.labels = np.asarray(self.labels)
        _, self.data = preprocess_and_add_noise(self.data, snr=snr)
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def preprocess_and_add_noise(x, snr, seed=0):
    scalers = StandardScaler().fit(x)
    x_std = scalers.transform(x)
    np.random.seed(seed)
    sigma_noise = np.sqrt(1. / snr)
    x_std_noisy = x_std + sigma_noise * np.random.randn(*x_std.shape)
    return x_std, x_std_noisy


ds_train = SyntheticDataset(
    n_samples=n_samples, lat_dim=true_lat_dims, n_feats=n_feats,
    n_classes=n_classes, train=True, snr=snr)
ds_val = SyntheticDataset(
    n_samples=n_samples, lat_dim=true_lat_dims, n_feats=n_feats,
    n_classes=n_classes, train=False, snr=snr)
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=batch_size, shuffle=True, num_workers=1)
        for x in ["train", "val"]}

method = manifold.TSNE(n_components=2, init="pca", random_state=0)
y_train = method.fit_transform(ds_train.data)
y_val = method.fit_transform(ds_val.data)
fig, axs = plt.subplots(nrows=3, ncols=2)
for cnt, (name, y, labels) in enumerate((
        ("train", y_train, ds_train.labels),
        ("val", y_val, ds_val.labels))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[0, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[0, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[0, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[0, cnt].set_title("GT clustering ({0})".format(name))
    axs[0, cnt].axis("tight")

#############################################################################
# ML clustering
# -------------
#
# As a ground truth we performed a K-means clustering of the data.

kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(ds_train.data)
train_labels = kmeans.labels_
train_acc = GMVAELoss.cluster_acc(train_labels, ds_train.labels)
print("-- K-Means ACC train", train_acc)
val_labels = kmeans.predict(ds_val.data)
val_acc = GMVAELoss.cluster_acc(val_labels, ds_val.labels)
print("-- K-Means ACC val",val_acc)

for cnt, (name, y, labels, acc) in enumerate((
        ("train", y_train, train_labels, train_acc),
        ("val", y_val, val_labels, val_acc))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[1, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[1, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[1, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[1, cnt].set_title(
        "K-means clustering ({0}-ACC:{1:.3f})".format(name, acc))
    axs[1, cnt].axis("tight")

#############################################################################
# Training
# --------
#
# Create/train the model.

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
                    criterion.layer_outputs = layer_outputs
                    loss, extra_loss = criterion(
                        outputs, batch_data, labels=batch_labels)
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

model = GMVAE(
    input_dim=n_feats, latent_dim=fit_lat_dims, n_mix_components=n_classes,
    sigma_min=0.001, raw_sigma_bias=0.25, dropout=0, temperature=1,
    gen_bias_init=0.)
print(model)
optimizer = optim.Adam(model.parameters(), lr=adam_lr)
criterion = GMVAELoss()
train_model(dataloaders, model, device, criterion, optimizer,
            scheduler=None, n_epochs=n_epochs, checkpointdir=None,
            board=None, load_best=False)

model.eval()
with torch.no_grad():
    p_x_given_z, dists = model(
        torch.from_numpy(ds_train.data.astype(np.float32)).to(device))
q_y_given_x = dists["q_y_given_x"]
train_labels = np.argmax(q_y_given_x.logits.detach().cpu().numpy(), axis=1)
train_acc = GMVAELoss.cluster_acc(
    q_y_given_x.logits, ds_train.labels, is_logits=True)
print("-- GMVAE ACC train", train_acc)
with torch.no_grad():
    p_x_given_z, dists = model(
            torch.from_numpy(ds_val.data.astype(np.float32)).to(device))
q_y_given_x = dists["q_y_given_x"]
val_labels = np.argmax(q_y_given_x.logits.detach().cpu().numpy(), axis=1)
val_acc = GMVAELoss.cluster_acc(
    q_y_given_x.logits, ds_val.labels, is_logits=True)
print("-- GMVAE ACC val", val_acc)

for cnt, (name, y, labels, acc) in enumerate((
        ("train", y_train, train_labels, train_acc),
        ("val", y_val, val_labels, val_acc))):
    colors = labels.astype(float)
    colors /= colors.max()
    axs[2, cnt].scatter(y[:, 0], y[:, 1], c=colors, cmap=plt.cm.Spectral)
    axs[2, cnt].xaxis.set_major_formatter(NullFormatter())
    axs[2, cnt].yaxis.set_major_formatter(NullFormatter())
    axs[2, cnt].set_title(
        "GMVAE clustering ({0}-ACC:{1:.3f})".format(name, acc))
    axs[2, cnt].axis("tight")
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                    wspace=0.1, hspace=0.5)
plt.show()
