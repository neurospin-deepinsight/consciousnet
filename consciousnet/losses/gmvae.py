# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Definition of the Gaussian Mixture Variational Auto-Encoder (GMVAE) losses.
"""

# Imports
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as func


class GMVAELoss(object):
    """ GMVAE Loss.
    """
    def __init__(self):
        """ Init class.
        """
        super(GMVAELoss, self).__init__()
        self.layer_outputs = None

    def __call__(self, p_x_given_z, data, labels=None):
        """ Compute loss.
        """
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        q_y_given_x = self.layer_outputs["q_y_given_x"]
        y = self.layer_outputs["y"]
        q_z_given_xy = self.layer_outputs["q_z_given_xy"]
        z = self.layer_outputs["z"]
        p_z_given_y = self.layer_outputs["p_z_given_y"]

        # Reconstruction loss term i.e. the negative log-likelihood
        nll = - p_x_given_z.log_prob(data).sum()
        nll /= len(data)

        # Latent loss between approximate posterior and prior for z
        kl_div_z = (q_z_given_xy.log_prob(z) - p_z_given_y.log_prob(z)).sum()
        kl_div_z /= len(data)

        # Conditional entropy loss
        logits = q_y_given_x.logits
        probs = func.softmax(logits, dim=-1)
        nent = (- probs * torch.log(probs)).sum()
        nent /= len(data)

        # Need to maximise the ELBO with respect to these weights
        loss = nll + kl_div_z + nent

        # Keep track of the clustering accuracy during training
        if labels is not None:
            cluster_acc = GMVAELoss.cluster_acc(
                q_y_given_x.logits, labels, is_logits=True)
        else:
            cluster_acc = 0

        return loss, {"nll": nll, "kl_div_z": kl_div_z, "nent": nent,
                      "cluster_acc": cluster_acc}

    @staticmethod
    def cluster_acc(y_pred, y, is_logits=False):
        # assert y_pred.size == y.size
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if is_logits:
            y_pred = np.argmax(y_pred, axis=1)
        n_classes = max(y_pred.max(), y.max()) + 1
        gain = np.zeros((n_classes, n_classes), dtype=np.int64)
        for idx in range(y_pred.size):
            gain[y_pred[idx], y[idx]] += 1
        cost = gain.max() - gain
        row_ind, col_ind = linear_sum_assignment(cost)
        return gain[row_ind, col_ind].sum() / y_pred.size


class VAEGMPLoss(object):
    """ VAEGMP Loss.
    """
    def __init__(self, beta=1., reduction="entropy"):
        """ Init class.

        Parameters
        ----------
        beta: float, default 1
            the weight of KL term regularization.
        reduction: str, default 'entropy'
            how to reduce the loss.
        """
        super(VAEGMPLoss, self).__init__()
        self.layer_outputs = None
        self.beta = beta
        self.reduction = reduction

    def __call__(self, p_x_given_z, data):
        """ Compute loss.
        """
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        q_z_given_x = self.layer_outputs["q_z_given_x"]
        z = self.layer_outputs["z"]
        p_z = self.layer_outputs["p_z"]

        # Reconstruction loss term i.e. the negative log-likelihood
        nll = - p_x_given_z.log_prob(data)

        # Latent loss between approximate posterior and prior for z
        kl_div_z = q_z_given_x.log_prob(z) - p_z.log_prob(z)

        # Reduction
        if self.reduction == "entropy":
            nll = nll.sum() / len(data)
            kl_div_z = kl_div_z.sum() / len(data)

        # Need to maximise the ELBO with respect to these weights
        loss = nll + self.beta * kl_div_z

        return loss, {"nll": nll, "kl_div_z": kl_div_z}
