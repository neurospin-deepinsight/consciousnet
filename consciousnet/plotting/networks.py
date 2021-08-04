# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2021
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Network plots.
"""

# Imports
import os
import json
import inspect
import textwrap
import importlib
import numpy as np
import matplotlib.pyplot as plt


def plot_circo(adj, names, hemi_groups, group_names, group_colors,
               outdir, with_labels=False):
    """ Generate a circular flow charts plot using singularity and graph-tools.

    Parameters
    ----------
    adj: array (N, N)
        the adjacency matrix.
    names: list of str (N, )
        the node names.
    hemi_groups: list of int (N, )
        the hemisphere tag labels.
    group_names: list of str (N, )
        the nodes associated group names.
    group_colors: list of 3-uplet
        the nodes associated group colors.
    outdir: str
        the destination folder.
    with_labels: bool, default False
        optionnaly display the group names.

    Returns
    -------
    circo_file: str
        the generated chart.
    """
    simg_file = get_graphtools_simg(outdir)
    return plot_chart(
        "generate_circo", outdir, simg_file, adj=adj, names=names,
        hemi_groups=hemi_groups, group_names=group_names,
        group_colors=group_colors, with_labels=with_labels)


def plot_graph(adj, names, hemi_groups, outdir):
    """ Display a graph plot using singularity and graph-tools.

    Parameters
    ----------
    adj: array (N, N)
        the adjacency matrix.
    names: list of str (N, )
        the node names.
    hemi_groups: list of int (N, )
        the hemisphere tag labels.
    outdir: str
        the destination folder.

    Returns
    -------
    graph_file: str
        the generated chart.
    """
    simg_file = get_graphtools_simg(outdir)
    return plot_chart(
        "generate_graph", outdir, simg_file, adj=adj, names=names,
        hemi_groups=hemi_groups)


def get_graphtools_simg(outdir):
    """ Retrieve a singularity image with graph-tools.

    Parameters
    ----------
    outdir: str
        the folder where the image will be saved.

    Returns
    -------
    simg_file: str
        the path to the image.
    """
    url = "http://biodev.cea.fr/singularity/graphtools-latest.simg"
    simg_file = os.path.join(outdir, "graphtools-latest.simg")
    if not os.path.isfile(simg_file):
        cmd = "wget -O {0} {1}".format(simg_file, url)
        os.system(cmd)
    return simg_file


def generate_circo(adj, names, hemi_groups, group_names, group_colors,
                   outfile, with_labels=True):
    """ Generate a circular flow charts plot using singularity and graph-tools.
    """
    import numpy as np
    import graph_tool.all as gt

    adj = np.asarray(adj)
    idx = adj.nonzero()
    weights = adj[idx]
    g = gt.Graph(directed=False)
    g.add_edge_list(np.transpose(idx))

    ew = g.new_edge_property("double")
    ew.a = weights
    g.ep["edge_weight"] = ew

    node_name = g.new_vertex_property("string", vals=group_names)
    g.vp["node_weight"] = node_name

    node_full_names = g.new_vertex_property("string", vals=names)
    g.vp["node_full_name"] = node_full_names

    node_colors = g.new_vertex_property("vector<float>", vals=group_colors)
    g.vp["node_colour"] = node_colors

    node_hemispheres = g.new_vertex_property("int", vals=hemi_groups)
    g.vp["hemi_group"] = node_hemispheres

    pos = gt.arf_layout(g, weight=ew, a=0.5, d=10)

    state = gt.minimize_nested_blockmodel_dl(g)
    kwargs = {}
    if with_labels:
        kwargs["vertex_text"] = node_name
    gt.draw_hierarchy(
        state, vertex_fill_color=node_colors, edge_pen_width=ew,
        output=outfile, layout="radial", rel_order=node_name,
        vertex_pen_width=1, vertex_size=15, vertex_text_color="#000000",
        vertex_font_size=7, vertex_font_family="helvetica",
        bg_color="#FFFFFF", deg_size=False, output_size=(1000, 1000),
        **kwargs)


def generate_graph(adj, names, hemi_groups, outfile):
    """ Display a graph plot using singularity and graph-tools.
    """
    import numpy as np
    import graph_tool.all as gt

    adj = np.asarray(adj)
    adj[np.tril_indices(len(adj))] = 0
    idx = adj.nonzero()
    weights = adj[idx]
    g = gt.Graph(directed=False)
    g.add_edge_list(np.transpose(idx))

    ew = g.new_edge_property("double")
    ew.a = weights
    g.ep["edge_weight"] = ew

    node_name = g.new_vertex_property("string", vals=names)
    g.vp["node_weight"] = node_name

    node_full_names = g.new_vertex_property("string", vals=names)
    g.vp["node_full_name"] = node_full_names

    node_hemispheres = g.new_vertex_property("int", vals=hemi_groups)
    g.vp["hemi_group"] = node_hemispheres

    # pos = gt.arf_layout(g, weight=ew, a=0.5, d=10)
    # pos = gt.fruchterman_reingold_layout(g, n_iter=1000, weight=ew)
    pos = gt.sfdp_layout(g, eweight=ew, p=15, groups=node_hemispheres, C=3)
    # pos = gt.random_layout(g)

    gt.graph_draw(
        g, pos=pos, vertex_font_size=10, vertex_shape="double_circle",
        vertex_fill_color="#729fcf", vertex_pen_width=3, vertex_text=node_name,
        output_size=(1000, 1000), edge_pen_width=ew, output=outfile,
        inline=True, bg_color="#FFFFFF")


def plot_chart(chart_name, outdir, simg_file, **kwargs):
    """ Generate a circular flow charts oor graph plots using singularity
    and graph-tools.
    """
    mod_name = "consciousnet.plotting." + inspect.getmodulename(__file__)
    mod = importlib.import_module(mod_name)
    funcs = dict(inspect.getmembers(mod, inspect.isfunction))
    func = funcs.get(chart_name, None)
    if func is None:
        raise ValueError("Unexpected function name '{0}'.".format(chart_name))
    outfile = os.path.join(
        outdir, chart_name.replace("generate_", "") + ".png")
    kwargs["outfile"] = outfile
    params = ""
    for key, val in kwargs.items():
        if val is None or isinstance(val, bool):
            params += "{0} = {1}\n".format(key, repr(val))
        else:
            params += "{0} = {1}\n".format(key, json.dumps(val))
    source = inspect.getsourcelines(func)[0]
    cut_idx = len(source) - source[::-1].index('    """\n')
    source = "".join(source[cut_idx:])
    source = textwrap.dedent(source)
    code = params + "\n" + source
    code_file = os.path.join(outdir, chart_name + ".py")
    with open(code_file, "wt") as of:
        of.write(code)
    cmd = "singularity exec --bind {0}:/out {1} python /out/{2}".format(
        outdir, simg_file, os.path.basename(code_file))
    print("Executing: ", cmd)
    os.system(cmd)
    return outfile
