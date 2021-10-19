from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, hmean
from sfacts.math import genotype_cdist, adjusted_community_dissimilarity
import pandas as pd
import numpy as np
import xarray as xr


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

    g = gammaA.shape[1]
    dist = pd.DataFrame(cdist(gammaA, gammaB))
    return (
        pd.Series(dist.idxmin(axis=1), index=reference.strain).rename_axis(
            index="strain"
        ),
        pd.Series(dist.min(axis=1), index=reference.strain).rename_axis(index="strain"),
    )


def genotype_error(reference, estimate, **kwargs):
    _, error = match_genotypes(reference, estimate, **kwargs)
    return error.mean(), error


def weighted_genotype_error(reference, estimate, weight_func=None, **kwargs):
    if weight_func is None:
        weight_func = lambda w: (w.data.mu * w.data.communities).sum("sample")

    weight = weight_func(reference)

    _, error = genotype_error(reference, estimate, **kwargs)
    return float((weight * error).sum() / weight.sum())


def community_error(reference, estimate):
    piA = reference.communities.to_pandas()
    piB = estimate.communities.to_pandas()
    bcA = squareform(pdist(piA, metric="braycurtis"))
    bcB = squareform(pdist(piB, metric="braycurtis"))

    out = []
    for i in range(len(bcA)):
        bcA_i, bcB_i = bcA[:, i], bcB[:, i]
        out.append(_mae(np.delete(bcA_i, i), np.delete(bcB_i, i)))

    return np.mean(out), pd.Series(out, index=reference.sample).rename_axis(
        index="sample"
    )


def max_strain_abundance_error(reference, estimate):
    sample_err = np.abs(
        reference.communities.max("strain") - estimate.communities.max("strain")
    )
    return float(sample_err.mean()), sample_err


def integrated_community_error(reference, estimate):
    trans_genotypes_cdist = genotype_cdist(
        reference.genotypes.values, estimate.genotypes.values
    )
    cis_genotypes_cdist = genotype_cdist(
        reference.genotypes.values, reference.genotypes.values
    )
    trans_outer = np.einsum(
        "ab,ac->abc", reference.communities.values, estimate.communities.values
    )
    cis_outer = np.einsum(
        "ab,ac->abc", reference.communities.values, reference.communities.values
    )
    trans_expect = (trans_outer * trans_genotypes_cdist).sum(axis=(1, 2))
    cis_expect = (cis_outer * cis_genotypes_cdist).sum(axis=(1, 2))

    err = np.abs(trans_expect - cis_expect)
    return np.mean(err), pd.Series(err, index=reference.sample).rename_axis(
        index="sample"
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


def community_error_test(reference, estimate, reps=99):
    pi_sim = reference.communities.to_pandas()
    pi_fit = estimate.communities.to_pandas()

    bc_sim = 1 - pdist(pi_sim, metric="braycurtis")
    bc_fit = 1 - pdist(pi_fit, metric="braycurtis")
    err = _mae(bc_sim, bc_fit)

    null = []
    # n = len(bc_sim)
    for i in range(reps):
        bc_sim_permute = np.random.permutation(bc_sim)
        null.append(_mae(bc_sim, bc_sim_permute))
    null = np.array(null)

    return err, null, err / np.mean(null), (np.sort(null) < err).mean()


def metagenotype_error(reference, estimate):
    estimated_metagenotypes = estimate.data.p * reference.data.m
    err = np.abs(estimated_metagenotypes - reference.metagenotypes.sel(allele="alt"))
    mean_sample_error = err.mean("position") / reference.data.mu
    return float(mean_sample_error.mean()), mean_sample_error.to_series()


def rank_abundance_error(reference, estimate):
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
            _mae(
                np.sort(reference_padded[i]),
                np.sort(estimate_padded[i]),
            )
        )

    return np.mean(err), pd.Series(err, index=reference.sample).rename_axis(
        index="sample"
    )
