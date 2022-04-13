from scipy.spatial.distance import squareform
from scipy.stats import hmean
from sfacts.math import (
    genotype_cdist,
    entropy,
)
from sfacts.unifrac import (
    neighbor_joining,
    unifrac_pdist,
)
from sfacts.data import Genotypes
import pandas as pd
import numpy as np


def _rmse(x, y):
    return np.sqrt(np.square(x - y).mean())


def _rss(x, y):
    return np.sqrt(np.square(x - y).sum())


def _mae(x, y):
    return np.abs(x - y).mean()


def _hmae(x, y):
    return hmean(np.abs(x - y))


def match_genotypes(reference, estimate, cdist=None):
    if cdist is None:
        cdist = genotype_cdist
    gammaA = reference.genotypes.data.to_pandas()
    gammaB = estimate.genotypes.data.to_pandas()
    dist = pd.DataFrame(cdist(gammaA, gammaB), index=gammaA.index, columns=gammaB.index)
    return (dist.idxmin(axis=1), dist.min(axis=1))


def discretized_match_genotypes(reference, estimate, cdist=None):
    if cdist is None:
        cdist = genotype_cdist
    gammaA = reference.genotypes.discretized().data.to_pandas()
    gammaB = estimate.genotypes.discretized().data.to_pandas()
    dist = pd.DataFrame(cdist(gammaA, gammaB), index=gammaA.index, columns=gammaB.index)
    return (dist.idxmin(axis=1), dist.min(axis=1))


def genotype_error(reference, estimate, **kwargs):
    _, error = match_genotypes(reference, estimate, **kwargs)
    return error.mean(), error


def discretized_genotype_error(reference, estimate, **kwargs):
    _, error = discretized_match_genotypes(reference, estimate, **kwargs)
    return error.mean(), error


def weighted_genotype_error(reference, estimate, weight_func=None, **kwargs):
    if weight_func is None:
        weight_func = lambda w: (
            w.metagenotypes.mean("sample") * w.data.communities
        ).sum("sample")

    weight = weight_func(reference)

    _, error = genotype_error(reference, estimate, **kwargs)
    return float((weight * error).sum() / weight.sum()), error


def discretized_weighted_genotype_error(
    reference, estimate, weight_func=None, **kwargs
):
    if weight_func is None:
        weight_func = lambda w: (w.metagenotypes.mean_depth() * w.data.communities).sum(
            "sample"
        )

    weight = weight_func(reference)

    _, error = discretized_genotype_error(reference, estimate, **kwargs)
    return float((weight * error).sum() / weight.sum()), error


def braycurtis_error(reference, estimate):
    bcA = reference.communities.pdist("sample").values
    bcB = estimate.communities.pdist("sample").values

    out = []
    for i in range(len(bcA)):
        bcA_i, bcB_i = bcA[:, i], bcB[:, i]
        out.append(_rmse(np.delete(bcA_i, i), np.delete(bcB_i, i)))

    return (
        np.mean(out),
        pd.Series(out, index=reference.sample).rename_axis(index="sample"),
    )


def max_strain_abundance_error(reference, estimate):
    sample_err = np.abs(
        reference.communities.max("strain") - estimate.communities.max("strain")
    )
    return float(sample_err.mean()), sample_err


def integrated_community_error(reference, estimate):
    est_genotypes_cdist = genotype_cdist(
        estimate.genotypes.values, estimate.genotypes.values
    )
    ref_genotypes_cdist = genotype_cdist(
        reference.genotypes.values, reference.genotypes.values
    )
    est_outer = np.einsum(
        "ab,ac->abc", estimate.communities.values, estimate.communities.values
    )
    ref_outer = np.einsum(
        "ab,ac->abc", reference.communities.values, reference.communities.values
    )
    est_expect = (est_outer * est_genotypes_cdist).sum(axis=(1, 2))
    ref_expect = (ref_outer * ref_genotypes_cdist).sum(axis=(1, 2))

    err = np.abs(est_expect - ref_expect)
    return (
        np.mean(err),
        pd.Series(err, index=reference.sample).rename_axis(index="sample"),
    )


def community_entropy_error(reference, estimate):
    ref_community_entropy = pd.Series(
        entropy(reference.communities.values), index=reference.sample
    ).rename_axis(index="sample")
    est_community_entropy = pd.Series(
        entropy(estimate.communities.values), index=reference.sample
    ).rename_axis(index="sample")
    diff = est_community_entropy - ref_community_entropy
    return (
        np.mean(np.abs(diff)),
        diff,
    )


def matched_strain_total_abundance_error(reference, estimate):
    best_match, _ = match_genotypes(reference, estimate, flip=True)
    out = np.empty_like(reference.communities.values)
    for i, _ in enumerate(reference.sample):
        for j, _ in enumerate(reference.strain):
            ref_abund = reference.communities.values[i, j]
            est_abund = estimate.communities.values[i, best_match == j]
            out[i, j] = np.abs(ref_abund - est_abund.sum())
    out = pd.DataFrame(out, index=reference.sample, columns=reference.strain)
    return out, out.sum(1), out.sum(1).mean()


def metagenotype_error(reference, estimate):
    estimated_metagenotypes = estimate.data.p * reference.data.m
    err = np.abs(estimated_metagenotypes - reference.metagenotypes.sel(allele="alt"))
    mean_sample_error = err.mean("position") / reference.data.mu
    return float(err.sum() / reference.data.m.sum()), mean_sample_error.to_series()


def metagenotype_error2(world, metagenotypes=None, discretized=False):
    if metagenotypes is None:
        metagenotypes = world.metagenotypes
    if discretized:
        g = world.genotypes.discretized().data
    else:
        g = world.genotypes.data
    p = world.data["communities"].data @ g.values
    m = metagenotypes.total_counts()
    mu = m.mean("position")
    x = metagenotypes.data.sel(allele="alt")
    predict = p * m
    err = np.abs(predict - x)
    mean_sample_error = err.mean("position") / mu
    return float(err.sum() / m.sum()), mean_sample_error.to_series()


def rank_abundance_error(reference, estimate, p=1):
    reference_num_strains = len(reference.strain)
    estimate_num_strains = len(estimate.strain)
    num_strains = max(reference_num_strains, estimate_num_strains)
    reference_padded = np.pad(
        reference.communities.values, (0, num_strains - reference_num_strains)
    )
    estimate_padded = np.pad(
        estimate.communities.values, (0, num_strains - estimate_num_strains)
    )

    err = []
    for i in range(len(reference.sample)):
        err.append(
            _mae(np.sort(reference_padded[i] ** p), np.sort(estimate_padded[i] ** p))
            ** (1 / p)
        )

    return (
        np.mean(err),
        pd.Series(err, index=reference.sample).rename_axis(index="sample"),
    )


def unifrac_error(reference, estimate, coef=1e6, discretized=False):
    from skbio import DistanceMatrix
    from skbio.diversity.beta import weighted_unifrac

    concat_genotypes = Genotypes.concat(
        {"ref": reference.genotypes, "est": estimate.genotypes}, dim="strain"
    )
    if discretized:
        concat_genotypes = concat_genotypes.discretized()

    dm = concat_genotypes.pdist()
    dm = DistanceMatrix(dm, dm.index.astype(str))
    tree = neighbor_joining(dm).root_at_midpoint()

    ref_com_stack = np.pad(
        reference.communities.values, pad_width=((0, 0), (0, estimate.sizes["strain"]))
    )
    est_com_stack = np.pad(
        estimate.communities.values, pad_width=((0, 0), (reference.sizes["strain"], 0))
    )
    diss = []
    for i in range(reference.sizes["sample"]):
        diss.append(
            weighted_unifrac(
                ref_com_stack[i] * coef,
                est_com_stack[i] * coef,
                otu_ids=concat_genotypes.strain.astype(str).values,
                tree=tree,
                validate=False,
            )
        )
    return (
        np.mean(diss),
        pd.Series(diss, index=reference.sample).rename_axis(index="sample"),
    )


def unifrac_error2(reference, estimate, coef=1e6, discretized=True):
    ref_pdist = squareform(unifrac_pdist(reference, coef=coef, discretized=discretized))
    est_pdist = squareform(unifrac_pdist(estimate, coef=coef, discretized=discretized))

    out = []
    for i in range(len(ref_pdist)):
        ref_i, est_i = ref_pdist[:, i], est_pdist[:, i]
        out.append(_mae(np.delete(ref_i, i), np.delete(est_i, i)))

    return (
        np.mean(out),
        pd.Series(out, index=reference.sample).rename_axis(index="sample"),
    )
