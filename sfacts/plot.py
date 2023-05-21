import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import xarray as xr
import warnings
import sfacts as sf


def _calculate_clustermap_sizes(
    nx,
    ny,
    n_row_colors=0,
    n_col_colors=0,
    scalex=0.15,
    scaley=0.02,
    cwidth=0,
    cheight=0,
    dwidth=0.2,
    dheight=1.0,
    pad_width=0.0,
    pad_height=0.0,
):
    # TODO: Incorporate colors.
    mwidth = nx * scalex
    mheight = ny * scaley
    fwidth = mwidth + cwidth + dwidth + pad_width
    fheight = mheight + cheight * n_col_colors + dheight * n_row_colors + pad_height
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
    col_cluster=True,
    row_cluster=True,
    col_linkage_func=None,
    row_linkage_func=None,
    row_col_annotation_cmap=mpl.cm.viridis,
    scalex=0.05,
    scaley=0.05,
    cwidth=0.2,
    cheight=0.2,
    dwidth=1.0,
    dheight=1.0,
    pad_width=0.5,
    pad_height=0.5,
    vmin=None,
    vmax=None,
    center=None,
    cmap=None,
    norm=mpl.colors.PowerNorm(1.0),
    xticklabels=0,
    yticklabels=0,
    metric="correlation",
    cbar_pos=None,
    transpose=False,
    background_color="darkgrey",
):
    def _plot_func(
        world,
        matrix_func=matrix_func,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
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
        center=center,
        cmap=cmap,
        norm=norm,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        metric=metric,
        cbar_pos=cbar_pos,
        transpose=transpose,
        pad_width=pad_width,
        pad_height=pad_height,
        background_color=background_color,
        **kwargs,
    ):

        if isinstance(world, sf.data.WrappedDataArrayMixin):
            world = world.to_world()

        matrix_data = matrix_func(world)

        matrix_data = matrix_data.to_pandas()

        if transpose:
            matrix_data = matrix_data.T
            col_linkage_func, row_linkage_func = (
                row_linkage_func,
                col_linkage_func,
            )
            col_colors_func, row_colors_func = row_colors_func, col_colors_func
            scalex, scaley = scaley, scalex
            cwidth, cheight = cheight, cwidth
            dwidth, dheight = dheight, dwidth
            xticklabels, yticklabels = yticklabels, xticklabels

        if col_linkage_func is None:
            col_linkage = None
        elif col_cluster:
            try:
                col_linkage = col_linkage_func(world)
            except ValueError as err:
                warnings.warn(f"col_linkage calculation failed: {err}")
                col_linkage = None
                col_cluster = False
        else:
            col_linkage = None

        if row_linkage_func is None:
            row_linkage = None
        elif row_cluster:
            try:
                row_linkage = row_linkage_func(world)
            except ValueError as err:
                warnings.warn(f"row_linkage calculation failed: {err}")
                row_linkage = None
                row_cluster = False
        else:
            row_linkage = None

        if col_colors_func is None:
            col_colors = None
            n_col_colors = 0
        else:
            col_colors = (
                col_colors_func(world)
                .pipe(_scale_to_max_of_one)
                .to_dataframe()
                .applymap(row_col_annotation_cmap)
            )
            n_col_colors = col_colors.shape[1]

        if row_colors_func is None:
            row_colors = None
            n_row_colors = 0
        else:
            row_colors = (
                row_colors_func(world)
                .pipe(_scale_to_max_of_one)
                .to_dataframe()
                .applymap(row_col_annotation_cmap)
            )
            n_row_colors = row_colors.shape[1]

        ny, nx = matrix_data.shape
        figsize, dendrogram_ratio, colors_ratio = _calculate_clustermap_sizes(
            nx,
            ny,
            n_row_colors=n_row_colors,
            n_col_colors=n_col_colors,
            scalex=scalex,
            scaley=scaley,
            cwidth=cwidth,
            cheight=cheight,
            dwidth=dwidth,
            dheight=dheight,
            pad_width=pad_width,
            pad_height=pad_height,
        )

        clustermap_kwargs = dict(
            vmin=vmin,
            vmax=vmax,
            center=center,
            norm=norm,
            cmap=cmap,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            col_cluster=col_cluster,
            row_cluster=row_cluster,
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

        grid = sns.clustermap(matrix_data, **clustermap_kwargs)
        grid.ax_heatmap.set_facecolor(background_color)
        return grid

    return _plot_func


plot_metagenotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.metagenotype.alt_allele_fraction(pseudo=0.0).T,
    row_linkage_func=lambda w: w.metagenotype.linkage(dim="position"),
    col_linkage_func=lambda w: w.metagenotype.linkage(dim="sample"),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    center=0.5,
    cmap="coolwarm",
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w.metagenotype.sum("allele")
            .mean("position")
            .pipe(np.sqrt)
            .rename("mean_depth")
        )
    ),
    background_color="darkgrey",
)


plot_expected_fractions = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.data["p"].T,
    row_linkage_func=lambda w: w.metagenotype.linkage(dim="position"),
    col_linkage_func=lambda w: w.metagenotype.linkage(dim="sample"),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    center=0.5,
    cmap="coolwarm",
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w.metagenotype.sum("allele")
            .mean("position")
            .pipe(np.sqrt)
            .rename("mean_depth")
        )
    ),
)

plot_prediction_error = plot_generic_clustermap_factory(
    matrix_func=lambda w: (w.data["p"] - w.metagenotype.frequencies().sel(allele="alt"))
    .fillna(0)
    .T,
    row_linkage_func=lambda w: w.metagenotype.linkage(dim="position"),
    col_linkage_func=lambda w: w.metagenotype.linkage(dim="sample"),
    scalex=0.15,
    scaley=0.01,
    vmin=-1,
    vmax=1,
    center=0,
    norm=mpl.colors.PowerNorm(1),
    cmap="coolwarm",
    xticklabels=1,
    yticklabels=0,
)

plot_dominance = plot_generic_clustermap_factory(
    matrix_func=lambda w: (
        w.metagenotype.dominant_allele_fraction()
        * (1 - w.metagenotype.dominant_allele_fraction())
        * np.sqrt(w.metagenotype.total_counts())
    ).T,
    col_linkage_func=lambda w: w.metagenotype.linkage(dim="sample"),
    row_linkage_func=lambda w: w.metagenotype.linkage(dim="position"),
    metric="cosine",
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=None,
    cmap=sns.cm.rocket_r,
    norm=mpl.colors.PowerNorm(1 / 2),
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w.metagenotype.sum("allele")
            .mean("position")
            .pipe(np.sqrt)
            .rename("mean_depth")
        )
    ),
)

plot_depth = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.metagenotype.sum("allele").T,
    row_linkage_func=lambda w: w.metagenotype.linkage(dim="position"),
    col_linkage_func=lambda w: w.metagenotype.linkage(dim="sample"),
    scalex=0.15,
    scaley=0.01,
    vmin=0,
    vmax=1,
    center=0.5,
    cmap="gray",
    xticklabels=1,
    yticklabels=0,
    col_colors_func=(
        lambda w: (
            w.metagenotype.sum("allele")
            .mean("position")
            .pipe(np.sqrt)
            .rename("mean_depth")
        )
    ),
    norm=mpl.colors.SymLogNorm(linthresh=1.0, base=10),
)

plot_genotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.genotype,
    row_linkage_func=lambda w: w.genotype.linkage(dim="strain"),
    col_linkage_func=lambda w: w.genotype.linkage(dim="position"),
    scaley=0.20,
    scalex=0.01,
    vmin=0,
    center=0.5,
    vmax=1,
    cmap="coolwarm",
    yticklabels=1,
    xticklabels=0,
    # row_colors_func=(lambda w: (w.genotype.entropy())),
)

plot_masked_genotype = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.masked_genotype,
    row_linkage_func=lambda w: w.masked_genotype.linkage(dim="strain"),
    col_linkage_func=lambda w: w.genotype.linkage(dim="position"),
    scaley=0.20,
    scalex=0.01,
    vmin=0,
    vmax=1,
    cmap="coolwarm",
    yticklabels=1,
    xticklabels=0,
    row_colors_func=(lambda w: (w.genotype.entropy())),
)

plot_missing = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.missingness,
    row_linkage_func=lambda w: w.genotype.linkage(dim="strain"),
    col_linkage_func=lambda w: w.genotype.linkage(dim="position"),
    metric="cosine",
    scaley=0.20,
    scalex=0.01,
    vmin=0,
    vmax=1,
    cmap=None,
    yticklabels=1,
    xticklabels=0,
    row_colors_func=(lambda w: (1 - w.missingness.mean("position"))),
)

plot_community = plot_generic_clustermap_factory(
    matrix_func=lambda w: w.community.data.T,
    # row_linkage_func=lambda w: w.genotype.linkage(dim="strain"),
    row_linkage_func=lambda w: w.community.linkage(dim="strain"),
    col_linkage_func=lambda w: w.community.linkage(dim="sample"),
    row_colors_func=(lambda w: (w.community.sum("sample").pipe(np.sqrt))),
    metric="cosine",
    scaley=0.20,
    scalex=0.15,
    dwidth=1.0,
    vmin=0,
    vmax=1,
    cmap=None,
    norm=mpl.colors.PowerNorm(1 / 2),
    xticklabels=1,
    yticklabels=1,
)


def plot_loss_history(*args, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    traces = []
    trace_min = []
    for trace in args:
        traces.append(np.array(trace))
        trace_min.append(traces[-1].min())
    trace_min = np.min(trace_min)
    for i, trace in enumerate(traces):
        ax.plot(trace - trace_min, label=f"{trace[-1]:0.3e} ({i})")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    return ax


def nmds_ordination(dmat):
    init = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=1,
        dissimilarity="precomputed",
        n_jobs=1,
    ).fit_transform(dmat)
    nmds = MDS(
        n_components=2,
        metric=False,
        max_iter=3000,
        eps=1e-12,
        dissimilarity="precomputed",
        random_state=1,
        n_jobs=1,
        n_init=1,
    )
    ordin = nmds.fit_transform(dmat, init=init)

    ordin = pd.DataFrame(
        ordin,
        index=dmat.index,
        columns=[f"PC{i}" for i in np.arange(ordin.shape[1]) + 1],
    )
    return ordin


def ordination_plot(
    world,
    dmat_func,
    ordin_func=nmds_ordination,
    colors_func=None,
    sizes_func=None,
    xy=("PC1", "PC2"),
    ax=None,
    **kwargs,
):
    """Plot nMDS ordination with markers colored/shaped by metadata features."""
    x, y = xy
    dmat = dmat_func(world)

    if colors_func is None:
        colors = None
    else:
        colors = colors_func(world)

    if sizes_func is None:
        sizes = None
    else:
        sizes = sizes_func(world)

    ordin = ordin_func(dmat)

    scatter_kwargs = dict(
        c=colors,
        cmap="viridis",
        s=sizes,
        edgecolor="k",
        lw=0.2,
        alpha=0.8,
    )
    scatter_kwargs.update(kwargs)
    if ax is None:
        ax = plt.gca()
    ax.scatter(x=x, y=y, data=ordin, **scatter_kwargs)
    ax.set_xlabel(f"{x}")
    ax.set_ylabel(f"{y}")
    return ax, ordin


def plot_metagenotype_frequency_spectrum(
    world,
    sample,
    bins=None,
    ax=None,
    **kwargs,
):
    hist_kwargs = dict()
    hist_kwargs.update(kwargs)

    if not ax:
        fig, ax = plt.subplots()

    if bins is None:
        bins = np.linspace(0.5, 1.0, num=21)

    frequencies = world.metagenotype.dominant_allele_fraction()
    ax.hist(frequencies.sel(sample=sample), bins=bins, color="black", **hist_kwargs)
    ax.set_title(sample)
    ax.set_xlim(0.5, 1)
    return ax


def plot_metagenotype_frequency_spectrum_compare_samples(
    world,
    sample_list=None,
    show_dominant=False,
    show_predict=False,
    axwidth=2,
    axheight=1.5,
    bins=None,
    axs=None,
    **kwargs,
):
    if sample_list is None:
        sample_list = world.sample.values

    hist_kwargs = dict()
    hist_kwargs.update(kwargs)

    n = len(sample_list)
    if not axs:
        fig, axs = plt.subplots(
            n, n, figsize=(axwidth * n, axheight * n), sharex=True, sharey=True
        )
    axs = np.asarray(axs).reshape((n, n))

    if bins is None:
        bins = np.linspace(0.5, 1.0, num=21)

    frequencies = world.metagenotype.mlift("sel", sample=sample_list).frequencies()
    for sample_i, row in zip(sample_list, axs):
        for sample_j, ax in zip(sample_list, row):
            domfreq_ij = (
                frequencies.sel(sample=[sample_i, sample_j])
                .mean("sample")
                .max("allele")
            )
            ax.hist(domfreq_ij, bins=bins, color="black", **hist_kwargs)
            if show_predict:
                frequencies_predict = world.data["p"].sel(sample=sample_list)
                freq_predict_ij = frequencies_predict.sel(
                    sample=[sample_i, sample_j]
                ).mean("sample")
                domfreq_predict_ij = xr.where(
                    freq_predict_ij > 0.5, freq_predict_ij, 1 - freq_predict_ij
                )
                ax.hist(
                    domfreq_predict_ij,
                    bins=bins,
                    color="red",
                    **hist_kwargs,
                    histtype="step",
                )

    if show_dominant:
        max_frac = world.community.sel(sample=sample_list).max("strain")
        max_frac_complement = 1 - max_frac
        for i, sample in enumerate(sample_list):
            ax = axs[i, i]
            ax.axvline(
                max_frac.sel(sample=sample),
                linestyle="--",
                lw=1,
                color="darkblue",
            )
            ax.axvline(
                max_frac_complement.sel(sample=sample),
                linestyle="--",
                lw=1,
                color="darkred",
            )

    for i, sample in enumerate(sample_list):
        ax_left = axs[i, 0]
        ax_top = axs[0, i]
        ax_left.set_ylabel(sample)
        ax_top.set_title(sample)

    ax.set_xlim(0.5, 1)


def plot_metagenotype_frequency_spectrum_compare_worlds(
    worlds, sample, alpha=0.5, bins=None, ax=None
):
    if bins is None:
        bins = np.linspace(0.5, 1.0, num=21)
    if ax is None:
        ax = plt.gca()

    plot_hist = lambda w, label: ax.hist(
        w.metagenotype.mlift("sel", sample=[sample])
        .dominant_allele_fraction()
        .values.squeeze(),
        bins=bins,
        alpha=alpha,
        label=label,
    )
    for k, w in worlds.items():
        plot_hist(w, k)
    ax.set_title(sample)

    return ax


def plot_beta_diversity_comparison(
    worldA,
    worldB,
    **kwargs,
):
    cdmatA = squareform(worldA.community.pdist(dim="sample"))
    cdmatB = squareform(worldB.community.pdist(dim="sample"))

    return sns.jointplot("a", "b", data=pd.DataFrame(dict(a=cdmatA, b=cdmatB)))
