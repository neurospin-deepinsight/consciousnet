# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Definition of the Temporal Difference Variational Auto-Encoder (TD-VAE) loss.
"""

# Imports
import torch.nn.functional as func
from torch.distributions import kl_divergence


class TDVAELoss(object):
    """ TDVAE Loss function.

    This loss needs intermediate layers outputs.
    Use a callback function to set the 'layer_outputs' class parameter before
    each evaluation of the loss function.
    If you use an interface this parameter is updated automatically.
    """
    def __init__(self, obs_loss=None):
        self.layer_outputs = None
        self.obs_loss = obs_loss or func.binary_cross_entropy

    def get_params(self):
        if self.layer_outputs is None:
            raise ValueError(
                "This loss needs intermediate layers outputs. Please register "
                "an appropriate callback.")
        t2 = self.layer_outputs["t2"]
        q_S_t1_li_z = self.layer_outputs["q_S_t1_li_z"]
        p_B_t1_li_z = self.layer_outputs["p_B_t1_li_z"]
        p_B_t2_li_z = self.layer_outputs["p_B_t2_li_z"]
        p_T_t2_li_z = self.layer_outputs["p_T_t2_li_z"]
        t2_li_z = self.layer_outputs["t2_li_z"]
        return t2, q_S_t1_li_z, p_B_t1_li_z, p_B_t2_li_z, p_T_t2_li_z, t2_li_z

    def __call__(self, p_D_t2_x, x):
        """ Calculate the jumpy TD-VAE loss, which corresponds to equations
        (6) and (8) in the reference paper.
        """
        # Get params
        (t2, q_S_t1_li_z, p_B_t1_li_z, p_B_t2_li_z, p_T_t2_li_z,
         t2_li_z) = self.get_params()
        n_layers = len(q_S_t1_li_z)
        xt2 = x[:, t2, :]

        # KL divergence between z distribution at time t1 based on
        # variational distribution (inference model) and z distribution at
        # time t1 based on belief.
        kl_qs_pb_loss = 0
        for i in range(n_layers):
            kl_qs_pb_loss += kl_divergence(q_S_t1_li_z[i], p_B_t1_li_z[i])
        kl_qs_pb_loss = kl_qs_pb_loss.sum(dim=-1)

        # The following four terms estimate the KL divergence between the z
        # distribution at time t2 based on variational distribution
        # (inference model) and z distribution at time t2 based on transition.
        # In contrast with the above KL divergence for z distribution at time
        # t1, this KL divergence can not be calculated analytically because
        # the transition distribution depends on z_t1, which is sampled
        # after z_t2. Therefore, the KL divergence is estimated using samples.
        kl_shift_qb_pt_loss = 0
        for i in range(n_layers):
            # - state log probabilty at time t2 based on belief
            kl_shift_qb_pt_loss += p_B_t2_li_z[i].log_prob(t2_li_z[i])
            # - state log probabilty at time t2 based on transition
            kl_shift_qb_pt_loss -= p_T_t2_li_z[i].log_prob(t2_li_z[i])
        kl_shift_qb_pt_loss = kl_shift_qb_pt_loss.sum(dim=-1)

        # Observation log probability at time t2
        _loss = self.obs_loss(p_D_t2_x, xt2, reduction="none").sum(dim=-1)
        _loss_optimal = self.obs_loss(xt2, xt2, reduction="none").sum(dim=-1)
        obs_loss = _loss - _loss_optimal

        # Final loss
        loss = kl_qs_pb_loss + kl_shift_qb_pt_loss + obs_loss
        loss = loss.mean()

        return loss, {"kl_qs_pb_loss": kl_qs_pb_loss,
                      "kl_shift_qb_pt_loss": kl_shift_qb_pt_loss,
                      "obs_loss": obs_loss}
