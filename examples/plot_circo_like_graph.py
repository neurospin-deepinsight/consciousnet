# -*- coding: utf-8 -*-
"""
Create circo-like graphs
========================

Credit: A Grigis

All the plots are perfomred with graph-tools embeded in a Singularity
singularity container. Please install first Singularity.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from consciousnet.plotting import plot_circo, plot_graph


#############################################################################
# Connectivity dataset
# --------------------
#
# First generate a connectivty matrix using the ROIs defined in the MSDL
# template.


def threshold_adj(adj, threshold):
    # Get number of connections to filter
    n_nodes = len(adj)
    n_conn_to_filter = int((threshold / 100.) * (n_nodes * (n_nodes - 1) / 2))

    # For threshold operations, zero out lower triangle (including diagonal)
    adj[np.tril_indices(n_nodes)] = 0

    # Following code is similar to bctpy
    indices = np.where(adj)
    sorted_indices = np.argsort(adj[indices])[::-1]
    adj[(indices[0][sorted_indices][n_conn_to_filter:],
         indices[1][sorted_indices][n_conn_to_filter:])] = 0

    # Just to get a symmetrical matrix
    adj = adj + adj.T

    # Diagonals need connection of 1 for graph operations
    adj[np.diag_indices(n_nodes)] = 1.0

    return adj


def imshow(image_file):
    image = plt.imread(image_file)
    fig, ax = plt.subplots()
    im = ax.imshow(image)
    ax.axis("off")


tmpdir = "/tmp/circo"
if not os.path.isdir(tmpdir):
    os.mkdir(tmpdir)
atlas = datasets.fetch_atlas_msdl(data_dir=tmpdir)
atlas_filename = atlas["maps"]
labels = atlas["labels"]
networks = atlas["networks"]
networks = [elem.decode("utf-8") for elem in networks]
data = datasets.fetch_development_fmri(n_subjects=1, data_dir=tmpdir)
fmri_filenames = data.func[0]
masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True,
                         verbose=5)
time_series = masker.fit_transform(fmri_filenames, confounds=data.confounds)
correlation_measure = ConnectivityMeasure(kind="correlation")
correlation_matrix = correlation_measure.fit_transform([time_series])[0]
correlation_matrix = threshold_adj(np.abs(correlation_matrix), 50)
np.fill_diagonal(correlation_matrix, 0)
correlation_matrix[correlation_matrix > 0.55] = (
    correlation_matrix[correlation_matrix > 0.55] * 1.1)
correlation_matrix[correlation_matrix > 0.85] = (
    correlation_matrix[correlation_matrix > 0.85] * 2.5)
correlation_matrix /= correlation_matrix.max()


#############################################################################
# Connectivity display
# --------------------
#
# Now display the connectivity as a graph or a circular flow chart.

adj = correlation_matrix.tolist()
names = labels
hemi_map = {"R": 0, "L": 1}
hemi_groups = [hemi_map.get(elem[0], 2) for elem in names]
colors = [
    "#FFC020", "#64064", "#146432", "#3CDC3C", "#14DC3C", "#A08CB4",
    "#DC1414", "#DC3C14", "#DCB4DC", "#9696C8", "#B42878", "#234B32",
    "#141E8C", "#E18C8C", "#C8234B", "#50148C", "#B4DC8C", "#A06432",
    "#64190", "#196428", "#78643C", "#50A014", "#14B48C", "#4B327D",
    "#DC3CDC", "#7D64A0", "#8CDCDC", "#3C14DC", "#DCB48C", "#DC14A",
    "#14DCA0", "#8C148C", "#DC1464", "#464646"]
colors = [tuple(int(elem.lstrip("#")[i: i + 2], 16) / 255. for i in (0, 2, 4))
          for elem in colors]
color_map = dict((key, colors[cnt]) for cnt, key in enumerate(set(networks)))
group_names = networks
group_colors = [color_map[elem] for elem in group_names]
circo_file = plot_circo(
    adj=adj, names=names, hemi_groups=hemi_groups, group_names=group_names,
    group_colors=group_colors, outdir=tmpdir, with_labels=False)
imshow(circo_file)
print(circo_file)
circo_file = plot_circo(
    adj=adj, names=names, hemi_groups=hemi_groups, group_names=group_names,
    group_colors=group_colors, outdir=tmpdir, with_labels=True)
print(circo_file)
imshow(circo_file)
graph_file = plot_graph(
    adj=adj, names=names, hemi_groups=hemi_groups, outdir=tmpdir)
print(graph_file)
imshow(graph_file)

plt.show()

