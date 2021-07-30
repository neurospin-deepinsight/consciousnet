# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Reconstruction plots.
"""

# Imports
import matplotlib.pyplot as plt
from skimage.metrics import (
    structural_similarity, peak_signal_noise_ratio, normalized_root_mse)


def plot_reconstruction_error(data, rec_data, fig=None, outfile=None):
    """ Display the reconstruction error.

    Parameters
    ----------
    data: array (N, d)
        the reference data.
    rec_data: array (N, d)
        the reconstructed data.
    fig: Figure, default None
        a matplotlib figure.
    outfile: str, default None
        optionally specify a file to save the plot.

    Returns
    -------
    similarity: dict
        the generated similarity metrics.
    """
    similarity = {}
    for cnt, (name, func) in enumerate((("mse", normalized_root_mse),
                                        ("ssim", structural_similarity),
                                        ("psnr", peak_signal_noise_ratio))):
        metrics = []
        for idx in range(len(data)):
            kwargs = {}
            if name in ("ssim", "psnr"):
                kwargs["data_range"] = (
                    rec_data[idx].max() - rec_data[idx].min())
            metrics.append(func(data[idx], rec_data[idx], **kwargs))
        similarity[name] = metrics
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(3, 2, cnt * 2 + 1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(name)
        ax.hist(metrics, alpha=0.5, bins=20)
        ax = fig.add_subplot(3, 2, cnt * 2 + 2)
        ax.axes.xaxis.set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.boxplot(metrics, showfliers=False)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.85,
                        wspace=0.1, hspace=0.5)
    fig.suptitle("Reconstruction accuracy")
    if outfile is not None:
        fig.savefig(outfile)
    return similarity
