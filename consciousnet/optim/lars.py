# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Layer-wise Adaptive Rate Scaling or LARS optimization.
"""

# Imports
import torch
from torch import optim


class LARS(optim.Optimizer):
    """ Layer-wise Adaptive Rate Scaling, or LARS, is a large batch
    optimization technique. There are two notable differences between LARS
    and other adaptive algorithms such as Adam or RMSProp: first, LARS uses
    a separate learning rate for each layer and not for each weight.
    And second, the magnitude of the update is controlled with respect to
    the weight norm for better control of training speed.
    """
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue
                if (g["weight_decay_filter"] is None or
                        not g["weight_decay_filter"](p)):
                    dp = dp.add(p, alpha=g["weight_decay"])
                if (g["lars_adaptation_filter"] is None or
                        not g["lars_adaptation_filter"](p)):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.,
                        torch.where(update_norm > 0,
                                    (g["eta"] * param_norm / update_norm),
                                    one),
                        one)
                    dp = dp.mul(q)
                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])
