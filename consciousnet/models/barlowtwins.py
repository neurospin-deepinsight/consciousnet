# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Definition of the Self-Supervised Learning via Redundancy Reduction
(Barlow Twins) model.
"""

# Imports
import torch
from torch import nn


class BarlowTwins(nn.Module):
    """ Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
    """
    def __init__(self, model, fc_layer_name, fc_in_features, projector,
                 batch_size, lambd):
        """ Init class.

        Parameters
        ----------
        model: nn.model
            the classification network.
        fc_layer_name: str
            the name of the fully conencted layer that will be replaced during
            the optimization by a projection head.
        fc_in_features: int
            the fully connected input features dimension.
        projector: str
            the MLP layers projector definition of the form 120-120-120.
        batch_size: int
            the mini-batch size.
        lambd: float
            the weight applied on off-diagonal terms.
        """
        super(BarlowTwins, self).__init__()
        self.batch_size = batch_size
        self.lambd = lambd

        # Encoder
        self.backbone = model
        setattr(self.backbone, fc_layer_name, nn.Identity())

        # Projector
        sizes = [fc_in_features] + list(map(int, projector.split("-")))
        layers = []
        for idx in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[idx], sizes[idx + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[idx + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # Normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        """ The forward method.

        Parameters
        ----------
        y1, y2: Tensors
            the contrasted input data.

        Returns
        -------
        loss: float
            the summed cross-correlation matrix.
        """
        z1 = self.projector(listify(self.backbone(y1))[0])
        z2 = self.projector(listify(self.backbone(y2))[0])

        # Empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # Sum the cross-correlation matrix
        c.div_(self.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag

        return loss


def listify(item):
    """ Ensure that the input is a list or tuple.

    Parameters
    ----------
    item: object or list or tuple
        the input data.

    Returns
    -------
    out: list
        the liftify input data.
    """
    if isinstance(item, list) or isinstance(item, tuple):
        return item
    else:
        return [item]


def off_diagonal(x):
    """ Return a flattened view of the off-diagonal elements of a square
    matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
