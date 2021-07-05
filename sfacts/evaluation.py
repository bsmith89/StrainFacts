from scipy.spatial.distance import cdist, pdist
import pandas as pd
import numpy as np
import xarray as xr


# def binary_entropy(p):
#     q = 1 - p
#     ent = -(p * np.log2(p) + q * np.log2(q))
#     return ent


# def sum_binary_entropy(p, normalize=False, axis=None):
#     q = 1 - p
#     ent = np.sum(-(p * np.log2(p) + q * np.log2(q)), axis=axis)
#     if normalize:
#         ent = ent / p.shape[axis]
#     return ent


# def mean_masked_genotype_entropy(gamma, delta):
#     return (binary_entropy(gamma) * delta).mean(1)


# def sample_mean_masked_genotype_entropy(pi, gamma, delta):
#     return (pi @ mean_masked_genotype_entropy(gamma, delta).reshape((-1, 1))).squeeze()


def _rmse(x, y):
    return np.sqrt(np.square(x - y).mean())


def _rss(x, y):
    return np.sqrt(np.square(x - y).sum())


def match_genotypes(worldA, worldB):
    gammaA = worldA.genotypes.data.to_pandas()
    gammaB = worldB.genotypes.data.to_pandas()

    g = gammaA.shape[1]
    dist = pd.DataFrame(cdist(gammaA, gammaB, metric="cityblock"))
    return dist.idxmin(axis=1), dist.min(axis=1) / g


def weighted_genotype_error(worldA, worldB):
    _, accuracy = match_genotypes(worldA, worldB)
    error = xr.DataArray(accuracy, dims=("strain",), coords=dict(strain=worldA.strain))
    total_coverage = (worldA.data.mu * worldA.data.communities).sum("sample")
    return float((error * total_coverage).sum() / total_coverage.sum())


def community_error(worldA, worldB):
    piA = worldA.communities.to_pandas()
    piB = worldB.communities.to_pandas()
    bcA = 1 - pdist(piA, metric="braycurtis")
    bcB = 1 - pdist(piB, metric="braycurtis")
    return _rmse(bcA, bcB)


def community_error_test(worldA, worldB, reps=99):
    pi_sim = worldA.communities.to_pandas()
    pi_fit = worldB.communities.to_pandas()

    bc_sim = 1 - pdist(pi_sim, metric="braycurtis")
    bc_fit = 1 - pdist(pi_fit, metric="braycurtis")
    err = _rmse(bc_sim, bc_fit)

    null = []
    # n = len(bc_sim)
    for i in range(reps):
        bc_sim_permute = np.random.permutation(bc_sim)
        null.append(_rmse(bc_sim, bc_sim_permute))
    null = np.array(null)

    return err, null, err / np.mean(null), (np.sort(null) < err).mean()


# def metacommunity_composition_rss(worldA, worldB):
#     pi_sim = worldA.communities.to_dataframe()
#     pi_fit = worldB.communities.to_dataframe()
#     mean_sim = pi_sim.mean(0)
#     mean_fit = pi_fit.mean(0)
#     s_sim = mean_sim.shape[0]
#     s_fit = mean_fit.shape[0]
#     s = max(s_sim, s_fit)
#     mean_sim = np.sort(np.pad(mean_sim, pad_width=(0, s - s_sim)))
#     mean_fit = np.sort(np.pad(mean_fit, pad_width=(0, s - s_fit)))
#     return _rss(mean_sim, mean_fit)
