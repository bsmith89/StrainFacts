from scipy.spatial.distance import cdist, pdist
import pandas as pd
import numpy as np
import xarray as xr


def _rmse(x, y):
    return np.sqrt(np.square(x - y).mean())


def _rss(x, y):
    return np.sqrt(np.square(x - y).sum())


def _mae(x, y):
    return np.abs(x - y).mean()


def match_genotypes(reference, estimate):
    gammaA = reference.genotypes.data.to_pandas()
    gammaB = estimate.genotypes.data.to_pandas()

    g = gammaA.shape[1]
    dist = pd.DataFrame(cdist(gammaA, gammaB, metric="cityblock"))
    return dist.idxmin(axis=1), dist.min(axis=1) / g


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
