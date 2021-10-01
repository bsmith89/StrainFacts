from scipy.spatial.distance import cdist, pdist
from scipy.stats import pearsonr
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


def match_genotypes(reference, estimate, flip=False, cdist=None):
    if cdist is None:
        cdist = genotype_cdist

    gammaA = reference.genotypes.data.to_pandas()
    gammaB = estimate.genotypes.data.to_pandas()
    if flip:
        gammaA, gammaB = gammaB, gammaA

    g = gammaA.shape[1]
    dist = pd.DataFrame(cdist(gammaA, gammaB))
    return dist.idxmin(axis=1), dist.min(axis=1)


def weighted_genotype_error(reference, estimate):
    _, accuracy = match_genotypes(reference, estimate)
    error = xr.DataArray(
        accuracy, dims=("strain",), coords=dict(strain=reference.strain)
    )
    total_coverage = (reference.data.mu * reference.data.communities).sum("sample")
    return float((error * total_coverage).sum() / total_coverage.sum())


def community_error(reference, estimate):
    piA = reference.communities.to_pandas()
    piB = estimate.communities.to_pandas()
    bcA = 1 - pdist(piA, metric="braycurtis")
    bcB = 1 - pdist(piB, metric="braycurtis")
    return _mae(bcA, bcB)


def integrated_community_error(reference, estimate):
    reference_pdist = pdist(
        reference.communities.values,
        adjusted_community_dissimilarity,
        gdiss=reference.genotypes.pdist(),
    )
    estimate_pdist = pdist(
        estimate.communities.values,
        adjusted_community_dissimilarity,
        gdiss=estimate.genotypes.pdist(),
    )
    return 1 - pearsonr(reference_pdist, estimate_pdist)[0]


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


def naive_prediction_error(reference, estimate):
    depth = reference.metagenotypes.data.sum("allele")
    estimate_count_prediction = (
        estimate.communities.data @ estimate.genotypes.data
    ) * depth
    delta = estimate_count_prediction - reference.metagenotypes.data.sel(allele="alt")
    total_absolute_error = np.abs(delta).sum()
    return float(total_absolute_error / depth.sum())
