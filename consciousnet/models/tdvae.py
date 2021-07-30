# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Definition of the Temporal Difference Variational Auto-Encoder (TD-VAE) model.
"""

# Imports
import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class TDVAE(torch.nn.Module):
    """ Hierachical Temporal Difference Variational Auto-Encoder with
    jumpy predictions.

    Temporal Difference Variational Auto-Encoder, Karol Gregor, George
    Papamakarios, Frederic Besse, Lars Buesing and Theophane Weber,
    ICLR 2019, https://openreview.net/forum?id=S1x4ghC9tQ.

    First, let's first go through some definitions which would help
    understanding what is going on in the following code:

    Observation: the observated variable x.

    Belief: as the model is feed with a sequence of observations, x_t, the
    model updates its belief state b_t, through a LSTM network. It
    is a deterministic function of x_t. We call b_t the belief at
    time t instead of belief state.

    State: the latent hidden state variable z.
    """
    def __init__(self, x_dim, b_dim, z_dim, t, d, n_layers=2, n_lstm_layers=1,
                 preproc_dim=None, add_sigmoid=True):
        """ Init class.

        Parameters
        -----------
        x_dim: int
            the dimension of observed data.
        b_dim: int
            the belief code dimension.
        z_dim: int
            the dimension of latent space.
        t: int
            in jumpy state modeling, t1 can be chosen uniformly from the
            sequence U(1,t).
        d: int
            in jumpy state modeling, t2 − t1 can be chosen uniformly over
            some finite range U(1,d).
        n_layers: int, default 2
            the number of hierachical level in the model.
        n_lstm_layers: int, default 1
            the number of recurrent layers, eg setting this paramter to 2
            would mean stacking two LSTMs together to form a stacked LSTM,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results.
        preproc_dim: int, default None
            the dimension of preprocessed observations. If not specified no
            preprocessing is applied.
        add_sigmoid: bool, default True
            apply sigmoid activation fct to the decoder.
        """
        # Inheritance
        super(TDVAE, self).__init__()

        # Parameters
        self.x_dim = x_dim
        self.b_dim = b_dim
        self.z_dim = z_dim
        self.t = t
        self.d = d
        self.n_layers = n_layers
        self.n_lstm_layers = n_lstm_layers
        self.preproc_dim = preproc_dim

        # Input pre-process layer
        if self.preproc_dim is not None:
            self.preproc_x = PreprocBlock(
                input_size=x_dim, preproc_size=preproc_dim)
        else:
            self.preproc_dim = x_dim
            self.preproc_x = None

        # N layer LSTM for aggregating belief states
        self.lstm = nn.LSTM(
            input_size=self.preproc_dim, hidden_size=b_dim, batch_first=True,
            num_layers=n_lstm_layers)

        # N layer state model is used. Sampling is done by sampling
        # higher layer first.
        _b_to_z, _infer_z, _transition_z = [], [], []
        for idx in range(n_layers - 1, -1, -1):
            extra_dim = (z_dim if idx < (n_layers - 1) else 0)
            # - belief to state (b_to_z): this corresponds to the p_B
            #   distribution in the reference. Weights are shared across time
            #   but not across layers.
            _b_to_z.append(
                DBlock(
                    input_size=(b_dim + extra_dim),
                    hidden_size=50, output_size=z_dim))
            # - given belief and state at time t2, infer the state at time t1
            # (infer_z): this corresponds to the q_S distribution in the
            # reference.
            _infer_z.append(
                DBlock(
                    input_size=(b_dim + n_layers * z_dim + extra_dim),
                    hidden_size=50, output_size=z_dim))
            # - given the state at time t1, model state at time t2 through
            #   state transition (transition_z): this corresponds to the p_T
            #   distribution in the reference.
            _transition_z.append(
                DBlock(
                    input_size=(n_layers * z_dim + extra_dim),
                    hidden_size=50, output_size=z_dim))
        self.b_to_z = nn.ModuleList(_b_to_z)
        self.infer_z = nn.ModuleList(_infer_z)
        self.transition_z = nn.ModuleList(_transition_z)
        # - state to observation (z_to_x): this corresponds to the p_D
        #   distribution in the reference.
        self.z_to_x = Decoder(
            z_size=(n_layers * z_dim), hidden_size=200, x_size=x_dim,
            add_sigmoid=add_sigmoid)

    def reparameterize(self, q):
        """ The reparametrization trick.
        """
        if self.training:
            z = q.rsample()
        else:
            z = q.loc
        return z

    def forward(self, x):
        """ The forward method.
        """
        # - pre-process image x
        if self.preproc_x is not None:
            x = self.preproc_x(x)

        # - aggregate the belief b
        b, (h_n, c_n) = self.lstm(x)

        # - sample t1 and t2
        t1 = np.random.randint(0, self.t)
        t2 = t1 + np.random.randint(1, self.d + 1)

        # Because the loss is based on variational inference, we need to
        # draw samples from the variational distribution in order to estimate
        # the loss function.

        # - sample a state z at time t2 using the reparametralization trick
        #   in layer 2 & 1 respectively. The result state is obtained by
        #   concatenating results from layer 1 and layer 2.
        p_B_t2_li_z = []
        t2_li_z = []
        for i in range(self.n_layers):
            if i == 0:
                p_B_t2_li_z.append(self.b_to_z[i](
                    b[:, t2, :]))
            else:
                p_B_t2_li_z.append(self.b_to_z[i](
                    torch.cat((b[:, t2, :], t2_li_z[-1]), dim=-1)))
            t2_li_z.append(self.reparameterize(p_B_t2_li_z[-1]))
        t2_z = torch.cat(t2_li_z[::-1], dim=-1)

        # - sample a state at time t1: infer state at time t1 based on states
        #   at time t2. The result state is obtained by concatenating results
        #   from layer 1 and layer 2.
        q_S_t1_li_z = []
        t1_li_z = []
        for i in range(self.n_layers):
            if i == 0:
                q_S_t1_li_z.append(self.infer_z[i](
                    torch.cat((b[:, t1, :], t2_z), dim=-1)))
            else:
                q_S_t1_li_z.append(self.infer_z[i](
                    torch.cat((b[:, t1, :], t2_z, t1_li_z[-1]), dim=-1)))
            t1_li_z.append(self.reparameterize(q_S_t1_li_z[-1]))
        t1_z = torch.cat(t1_li_z[::-1], dim=-1)

        # - compute state distribution at time t1 based on belief at time 1
        p_B_t1_li_z = []
        for i in range(self.n_layers):
            if i == 0:
                p_B_t1_li_z.append(self.b_to_z[i](
                    b[:, t1, :]))
            else:
                p_B_t1_li_z.append(self.b_to_z[i](
                    torch.cat((b[:, t1, :], t1_li_z[i - 1]), dim=-1)))

        # - compute state distribution at time t2 based on states at time t1
        #   and state transition
        p_T_t2_li_z = []
        for i in range(self.n_layers):
            if i == 0:
                p_T_t2_li_z.append(self.transition_z[i](
                    t1_z))
            else:
                p_T_t2_li_z.append(self.transition_z[i](
                    torch.cat((t1_z, t2_li_z[i - 1]), dim=-1)))

        # - compute observation distribution at time t2 based on state at
        #   time t2
        p_D_t2_x = self.z_to_x(t2_z)

        return p_D_t2_x, {"b": b, "t1": t1, "t2": t2,
                          "p_B_t2_li_z": p_B_t2_li_z, "t2_li_z": t2_li_z,
                          "t2_z": t2_z, "q_S_t1_li_z": q_S_t1_li_z,
                          "t1_li_z": t1_li_z, "t1_z": t1_z,
                          "p_B_t1_li_z": p_B_t1_li_z,
                          "p_T_t2_li_z": p_T_t2_li_z}

    def rollout(self, x, t1, t2):
        """ Jumpy rollout.

        Parameters
        ----------
        x: Tensor
            the input sequences.
        t1: int
            the time jump number of steps.
        t2: int
            the prediction interval t2 - t1.

        Retruns
        -------
        rollout_x: Tensor
            the predicted frames.
        """
        # Compute belief
        if self.preproc_x is not None:
            x = self.preproc_x(x)
        b, (h_n, c_n) = self.lstm(x)

        # At time t1-1, we sample a state z based on belief at time t1-1
        li_z = []
        for i in range(self.n_layers):
            if i == 0:
                p_B_li_z = self.b_to_z[i](b[:, t1 - 1, :])
            else:
                p_B_li_z = self.b_to_z[i](
                    torch.cat((b[:, t1 - 1, :], li_z[-1]), dim=-1))
            li_z.append(self.reparameterize(p_B_li_z))
        current_z = torch.cat(li_z[::-1], dim=-1)

        # Start rollout
        rollout_x = []
        for k in range(t2 - t1 + 1):
            # - predict states after time t1 using state transition
            tnext_li_z = []
            for i in range(self.n_layers):
                if i == 0:
                    p_T_tnext_li_z = self.transition_z[i](current_z)
                else:
                    p_T_tnext_li_z = self.transition_z[i](
                        torch.cat((current_z, tnext_li_z[i - 1]), dim=-1))
                tnext_li_z.append(self.reparameterize(p_T_tnext_li_z))
            next_z = torch.cat(tnext_li_z[::-1], dim=-1)
            # - generate an observation x_t1 at time t1 based on sampled
            #   state z_t1
            next_x = self.z_to_x(next_z)
            rollout_x.append(next_x)
            current_z = next_z
        rollout_x = torch.stack(rollout_x, dim=1)

        return rollout_x


class DBlock(nn.Module):
    """ A basic building block to parametrize a Normal distribution.
    It is corresponding to the D operation in the reference Appendix.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_logsigma = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        t = torch.tanh(self.fc1(input))
        t = t * torch.sigmoid(self.fc2(input))
        mu = self.fc_mu(t)
        logsigma = self.fc_logsigma(t)
        return Normal(loc=mu, scale=logsigma.exp().pow(0.5))


class PreprocBlock(nn.Module):
    """ The optional pre-process layer.
    """
    def __init__(self, input_size, preproc_size):
        super(PreprocBlock, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, preproc_size)
        self.fc2 = nn.Linear(preproc_size, preproc_size)

    def forward(self, input):
        t = torch.relu(self.fc1(input))
        t = torch.relu(self.fc2(t))
        return t


class Decoder(nn.Module):
    """ The decoder layer that converts state to observation.

    Because the observation is MNIST image whose elements are values
    between 0 and 1, the output of this layer are probabilities of
    elements being 1.
    """
    def __init__(self, z_size, hidden_size, x_size, add_sigmoid=True):
        super(Decoder, self).__init__()
        self.add_sigmoid = add_sigmoid
        self.fc1 = nn.Linear(z_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, x_size)

    def forward(self, z):
        t = torch.tanh(self.fc1(z))
        t = torch.tanh(self.fc2(t))
        p = self.fc3(t)
        if self.add_sigmoid:
            p = torch.sigmoid(p)
        return p
