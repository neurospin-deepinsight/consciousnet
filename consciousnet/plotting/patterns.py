# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Spatio temporal patterns plots.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt


def plot_spatiotemporal_patterns(patterns, sigma, channel_id, fig=None,
                                 outfile=None):
    """ Display the spatiotemporal patterns.

    Parameters
    ----------
    patterns: array (l, s, c, d)
        the patterns to be displayed.
    sigma: float
        the traversal range [+sigma, -sigma].
    channel_id: int
        the channel to be displayed [0, c[.
    fig: Figure, default None
        a matplotlib figure.
    outfile: str, default None
        optionally specify a file to save the plot.
    """
    if fig is None:
        fig = plt.figure()
    fig = mosaic(
        patterns[:, :, channel_id], fig=fig,
        title=(r"Temporal patterns using [$-{0}\sigma$, $+{0}\sigma$] "
               "traversal").format(sigma),
        y_labels=[r"$-{0}\sigma$".format(sigma), r"0", r"$+{0}\sigma$".format(
            sigma)])
    if outfile is not None:
        fig.savefig(outfile)


def mosaic(data, title=None, y_labels=None, ncol=4, fig=None):
    """ Display a mosaic of images.
    """
    n_plots = len(data)
    nrow = n_plots // ncol
    vmin = data.min()
    vmax = np.percentile(data, 99)
    if n_plots % ncol != 0:
        nrow += 1
    if fig is None:
        fig = plt.figure()
    for idx, img in enumerate(data):
        ax = fig.add_subplot(nrow, ncol, idx + 1)
        ax.imshow(img, vmin=vmin, vmax=vmax, aspect="auto", cmap="jet")
        ax.set_xlim(0, img.shape[1])
        ax.set_xticks(np.arange(0, img.shape[1] + 1, img.shape[1] // 2))
        ax.set_ylim(0, img.shape[0])
        ax.set_yticks(np.arange(0, img.shape[0] + 1, img.shape[0] // 2))
        if idx % ncol != 0:
            ax.get_yaxis().set_visible(False)
        if idx < (n_plots - ncol) != 0:
            ax.get_xaxis().set_visible(False)
        if y_labels is not None:
            assert len(ax.get_xticklabels()) == len(y_labels)
            ax.set_yticklabels(y_labels, fontsize=8)
    for _idx in range(idx + 1, nrow * ncol):
        ax.axis("off")
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.87,
                        wspace=0.1, hspace=0.25)
    if title is not None:
        plt.suptitle(title)
