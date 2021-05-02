import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
from tqdm import tqdm
import sfacts as sf
from functools import partial


def _calculate_clustermap_sizes(
    nx, ny, scalex=0.15, scaley=0.02, cwidth=0, cheight=0, dwidth=0.2, dheight=1.0
):
    # TODO: Incorporate colors.
    mwidth = nx * scalex
    mheight = ny * scaley
    fwidth = mwidth + cwidth + dwidth
    fheight = mheight + cheight + dheight
    dendrogram_ratio = (dwidth / fwidth, dheight / fheight)
    colors_ratio = (cwidth / fwidth, cheight / fheight)
    return (fwidth, fheight), dendrogram_ratio, colors_ratio


def _min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def _scale_to_max_of_one(x):
    return x / x.max()


def dictionary_union(*args):
    out = args[0].copy()
    for a in args:
        out.update(a)
    return out


def plot_generic_clustermap_factory(
    matrix_func,
    col_colors_func=None,
    row_colors_func=None,
    col_linkage_func=None,
    row_linkage_func=None,
    row_col_annotation_cmap=mpl.cm.viridis,
    scalex=0.05,
    scaley=0.05,
    cwidth=0.4,
    cheight=0.4,
    dwidth=1.0,
    dheight=1.0,
    vmin=None,
    vmax=None,
    cmap=None,
    norm=mpl.colors.PowerNorm(1.),
    xticklabels=0,
    yticklabels=0,
    metric='correlation',
    cbar_pos=None,
):


    def _plot_func(
        world,
        matrix_func=matrix_func,
        col_linkage_func=col_linkage_func,
        row_linkage_func=row_linkage_func,
        col_colors_func=col_colors_func,
        row_colors_func=row_colors_func,
        row_col_annotation_cmap=row_col_annotation_cmap,
        scalex=scalex,
        scaley=scaley,
        cwidth=cwidth,
        cheight=cheight,
        dwidth=dwidth,
        dheight=dheight,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        norm=norm,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        metric=metric,
        cbar_pos=cbar_pos,
        **kwargs,
    ):
        if col_linkage_func is None:
            col_linkage = None
        else:
            col_linkage = col_linkage_func(world)

        if row_linkage_func is None:
            row_linkage = None
        else:
            row_linkage = row_linkage_func(world)
        
        if col_colors_func is None:
            col_colors = None
        else:
            col_colors = col_colors_func(world).pipe(_scale_to_max_of_one).to_dataframe().applymap(row_col_annotation_cmap)

        if row_colors_func is None:
            row_colors = None    
        else:
            row_colors = row_colors_func(world).pipe(_scale_to_max_of_one).to_dataframe().applymap(row_col_annotation_cmap)

        matrix_data = matrix_func(world)
        
        ny, nx = matrix_data.shape
        figsize, dendrogram_ratio, colors_ratio = _calculate_clustermap_sizes(
            nx,
            ny,
            scalex=scalex,
            scaley=scaley,
            cwidth=cwidth,
            cheight=cheight,
            dwidth=dwidth,
            dheight=dheight,
        )
    #     sf.logging_util.info(matrix_data.shape, applied_scale_kwargs, figsize, dendrogram_ratio, colors_ratio)
    
        clustermap_kwargs = dict(
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            cmap=cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            col_linkage=col_linkage,
            row_linkage=row_linkage,
            row_colors=row_colors,
            col_colors=col_colors,
            figsize=figsize,
            dendrogram_ratio=dendrogram_ratio,
            colors_ratio=colors_ratio,
            metric=metric,
            cbar_pos=cbar_pos,
        )
        clustermap_kwargs.update(kwargs)
        
        grid = sns.clustermap(
            matrix_data,
            **clustermap_kwargs
        )
        return grid
    return _plot_func
    

plot_metagenotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.metagenotypes.to_genotype_estimates().to_pandas().T,
    col_linkage_func=lambda w: w.metagenotypes.linkage(dim='strain', pseudo=1.),
    row_linkage_func=lambda w: w.metagenotypes.to_genotype_estimates(pseudo=1.).linkage(dim='sample'),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    cmap=mpl.cm.coolwarm,
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w
            .metagenotypes
            .sum('allele')
            .mean('position')
            .pipe(np.sqrt)
            .rename('mean_depth')
        )
    ),
)

plot_genotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.genotypes.to_pandas().T,
    col_linkage_func=lambda w: w.genotypes.linkage(dim='strain'),
    row_linkage_func=lambda w: w.genotypes.linkage(dim='position'),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    cmap=mpl.cm.coolwarm,
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w
            .genotypes
            .entropy
        )
    ),
)

plot_fuzzed_genotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.fuzzed_genotypes.to_pandas().T,
    col_linkage_func=lambda w: w.fuzzed_genotypes.linkage(dim='strain'),
    row_linkage_func=lambda w: w.genotypes.linkage(dim='position'),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    cmap=mpl.cm.coolwarm,
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w
            .genotypes
            .entropy
        )
    ),
)

plot_missing = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.missingness.to_pandas().T,
    col_linkage_func=lambda w: w.genotypes.linkage(dim='strain'),
    row_linkage_func=lambda w: w.genotypes.linkage(dim='position'),
    metric='cosine',
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    cmap=None,
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            1 - w
            .missingness
            .mean('position')
        )
    ),
)

plot_community = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.communities.to_pandas(),
    col_linkage_func=lambda w: w.genotypes.linkage(dim='strain'),
    metric='cosine',
    scalex=0.15,
    scaley=0.14,
    dheight=1.0,
    vmin=0,
    vmax=1,
    cmap=None,
    norm=mpl.colors.PowerNorm(1/2),
    xticklabels=1,
    yticklabels=1,
    col_colors_func=(
        lambda w: (
            w
            .communities
            .sum('sample')
            .pipe(np.sqrt)
        )
    ),
)

def plot_loss_history(trace):
    trace = np.array(trace)
    plt.plot((trace - trace.min()))
    plt.yscale("log")
