# -*- coding: utf-8 -*-
"""
Jumpy prediction with Temporal Difference Variational Auto-Encoder (TD-VAE)
===========================================================================

Credit: A Grigis

TD-VAE is designed such that it have all following three features:

* it learns a state representation of observations and makes predictions on
  the state level.
* based on observations, it learns a belief state that contains all
  the inforamtion required to make predictions about the future.
* it learns also to make predictions multiple steps in the future directly
  instead of make predictions step by step by connecting states
  that are multiple steps apart.

In this example we reproduce the experiment about moving MNIST digits.
In this experiment, a sequence of a MNIST digit moving to the left or the
right direction is presented to the model. The model need to predict how the
digit moves in the following steps. After training the model, a sequence of
digits can be fed into the model to see how well it can predict the further.

The `test` variable must be set to False to run a full training.
"""

import os
import sys
import time
import copy
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataify import MovingMNISTDataset
from consciousnet.models import TDVAE
from consciousnet.losses import TDVAELoss

test = True
datasetdir = "/tmp/moving_mnist"
if not os.path.isdir(datasetdir):
    os.mkdir(datasetdir)
input_size = 784
processed_x_size = 784
belief_state_size = 50
state_size = 8
t = 16
d = 4
add_sigmoid = True
n_samples = 3 if test else 512
n_epochs = 3 if test else 4000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################################################
# Moving MNIST digits dataset
# ---------------------------
#
# Fetch & load the moving MNIST digits dataset.

ds_train = MovingMNISTDataset(
    root=datasetdir, train=True, seq_size=20, shift=1, binary=True)
ds_val = MovingMNISTDataset(
    root=datasetdir, train=False, seq_size=20, shift=1, binary=True)
if test:
    ds_train = torch.utils.data.random_split(
        ds_train, [100, len(ds_train) - 100])[0]
    ds_val = torch.utils.data.random_split(
        ds_val, [100, len(ds_val) - 100])[0]
datasets = {"train": ds_train, "val": ds_val}
dataloaders = {x: torch.utils.data.DataLoader(
    datasets[x], batch_size=n_samples, shuffle=True, num_workers=1)
        for x in ["train", "val"]}

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
            for batch_data, _ in dataloaders[phase]:
                batch_data = batch_data.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward:
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs, layer_outputs = model(batch_data)
                    criterion.layer_outputs = layer_outputs
                    loss, extra_loss = criterion(outputs, batch_data)
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


model = TDVAE(x_dim=input_size, b_dim=belief_state_size, z_dim=state_size,
              t=t, d=d, n_layers=2, n_lstm_layers=1,
              preproc_dim=processed_x_size, add_sigmoid=add_sigmoid)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = TDVAELoss(obs_loss=torch.nn.functional.binary_cross_entropy)
train_model(dataloaders, model, device, criterion, optimizer,
            scheduler=None, n_epochs=n_epochs, checkpointdir=None,
            board=None, load_best=False)

#############################################################################
# Jumpy predictions
# ----------------
#
# A sequence of digits is fed into the model to see how well it can
# predict the 4 further images with a time jump of 11 steps.

t1, t2 = 11, 15
batch_display_size = 6
model.eval()
idx, (data, _) = next(enumerate(dataloaders["val"]))
data = data.to(device)
# calculate belief
model.forward(data)
# jumpy rollout
rollout_data = model.rollout(data, t1, t2)
# plot results
images = data.cpu().detach().numpy()
rollout_images = rollout_data.cpu().detach().numpy()
fig = plt.figure(0, figsize=(12, 4))
fig.clf()
gs = gridspec.GridSpec(batch_display_size, t2 + 2)
gs.update(wspace = 0.05, hspace = 0.05)
for i in range(batch_display_size):
    for j in range(t1):
        axes = plt.subplot(gs[i, j])
        axes.imshow(1 - images[i, j].reshape(28, 28),
                    cmap="binary")
        axes.axis("off")
    for j in range(t1, t2 + 1):
        axes = plt.subplot(gs[i, j + 1])
        axes.imshow(1 - rollout_images[i, j - t1].reshape(28, 28),
                    cmap="binary")
        axes.axis("off")
plt.show()

