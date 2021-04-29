import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
from tqdm import tqdm
import sfacts as sf


def _calculate_clustermap_sizes(
    nx, ny, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0
):
    # TODO: Incorporate colors.
    mwidth = nx * scalex
    mheight = ny * scaley
    fwidth = mwidth + dwidth
    fheight = mheight + dheight
    dendrogram_ratio = (dwidth / fwidth, dheight / fheight)
    return fwidth, fheight, dendrogram_ratio


def plot_genotype(
    point, linkage_kw=None, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0, **kwargs
):
    if linkage_kw is None:
        linkage_kw = {}
    linkage = point.genotypes.linkage(**linkage_kw)

    nx, ny = point.genotypes.data.shape
    fwidth, fheight, dendrogram_ratio = _calculate_clustermap_sizes(
        nx,
        ny,
        scalex=scalex,
        scaley=scaley,
        dwidth=dwidth,
        dheight=dheight,
    )

    kw = dict(
        vmin=0,
        vmax=1,
        cmap="coolwarm",
        dendrogram_ratio=dendrogram_ratio,
        col_linkage=linkage,
        figsize=(fwidth, fheight),
        xticklabels=1,
        yticklabels=0,
    )
    kw.update(kwargs)
    grid = sns.clustermap(point.genotypes.data.to_pandas().T, **kw)
    grid.cax.set_visible(False)
    return grid


def plot_missing(point, scalex=0.15, scaley=0.02, dwidth=0.2, dheight=1.0, **kwargs):
    nx, ny = point.missingness.data.shape
    fwidth, fheight, dendrogram_ratio = _calculate_clustermap_sizes(
        nx,
        ny,
        scalex=scalex,
        scaley=scaley,
        dwidth=dwidth,
        dheight=dheight,
    )

    kw = dict(
        vmin=0,
        vmax=1,
        dendrogram_ratio=dendrogram_ratio,
        figsize=(fwidth, fheight),
        xticklabels=1,
        yticklabels=0,
    )
    kw.update(kwargs)
    grid = sns.clustermap(point.missingness.data.to_pandas().T, **kw)
    grid.cax.set_visible(False)
    return grid


def plot_community(point, scalex=0.2, scaley=0.15, dwidth=0.2, dheight=1.0, **kwargs):
    ny, nx = point.communities.data.shape
    fwidth, fheight, dendrogram_ratio = _calculate_clustermap_sizes(
        nx,
        ny,
        scalex=scalex,
        scaley=scaley,
        dwidth=dwidth,
        dheight=dheight,
    )

    kw = dict(
        metric="cosine",
        vmin=0,
        vmax=1,
        dendrogram_ratio=dendrogram_ratio,
        figsize=(fwidth, fheight),
        xticklabels=1,
        yticklabels=1,
    )
    kw.update(kwargs)
    grid = sns.clustermap(point.communities.data.to_pandas(), **kw)
    grid.cax.set_visible(False)
    return grid


def plot_genotype_similarity(point, linkage_kw=None, **kwargs):
    if linkage_kw is None:
        linkage_kw = {}
    linkage = point.genotypes.linkage(**linkage_kw)
    dmat = point.genotypes.pdist()
    nx = ny = point.genotypes.shape[0]
    fwidth, fheight, dendrogram_ratio = _calculate_clustermap_sizes(
        nx, ny, scalex=0.15, scaley=0.15, dwidth=0.5, dheight=0.5
    )

    kw = dict(
        vmin=0,
        vmax=1,
        dendrogram_ratio=dendrogram_ratio,
        row_linkage=linkage,
        col_linkage=linkage,
        figsize=(fwidth, fheight),
        xticklabels=1,
        yticklabels=1,
    )
    kw.update(kwargs)
    grid = sns.clustermap(1 - dmat, **kw)
    grid.cax.set_visible(False)
    return grid


def plot_genotype_comparison(point_mapping, **kwargs):
    stacked = sf.data.Genotypes.stack({k: point_mapping[k].genotypes for k in point_mapping}, 'strain', prefix=True)
    kw = dict(xticklabels=1)
    kw.update(kwargs)
    return plot_genotype(stacked, **kw)


def plot_missing_comparison(point_mapping, **kwargs):
    stacked = sf.data.Missingness.stack({k: point_mapping[k].missingness for k in point_mapping}, 'strain', prefix=True)
    kw = dict(xticklabels=1)
    kw.update(kwargs)
    return plot_missing(stacked, **kw)


def plot_community_comparison(point_mapping, **kwargs):
    stacked = sf.data.Communities.stack({k: point_mapping[k].communities for k in point_mapping}, 'strain', prefix=True, validate=False)
    kw = dict(xticklabels=1)
    kw.update(kwargs)
    return plot_community(stacked, **kw)

# def plot_community_comparison(points, **kwargs):
#     stacked = pd.concat(
#         [p.communities for p in points]
#         axis=1,
#     )
#     kw = dict(xticklabels=1)
#     kw.update(kwargs)
#     return plot_community(stacked, **kw)


def plot_loss_history(trace):
    trace = np.array(trace)
    plt.plot((trace - trace.min()))
    plt.yscale("log")
