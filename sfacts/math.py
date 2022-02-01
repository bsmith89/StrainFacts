import numpy as np
from scipy.spatial.distance import cdist, pdist
import warnings


def binary_entropy(p):
    q = 1 - p
    return np.nan_to_num(-(p * np.log2(p) + q * np.log2(q)))


def entropy(p, axis=-1):
    return np.nan_to_num(-(p * np.log2(p)).sum(axis))


def genotype_binary_to_sign(p):
    return 2 * p - 1


def genotype_dissimilarity(x, y, q=2):
    "Dissimilarity between 1D genotypes, accounting for fuzzyness."
    x = genotype_binary_to_sign(x)
    y = genotype_binary_to_sign(y)

    dist = np.abs((x - y) / 2) ** q
    weight = np.abs(x * y)
    wmean_dist = ((weight * dist).sum()) / ((weight.sum()))
    # Why not finish up by powering it by (1 / q)?
    # I don't do this part because it loses the city-block distance
    # interpretation when x and y are both discrete (i.e. one of {0, 1}).

    # While the basic function is undefined where weight.sum() == 0
    # (and this is only true when one of x or y is always exactly 0.5 at every
    # index),
    # the limit approaches the same value from both directions.
    # We therefore redefine the dissimilarity as a piecewise function,
    # but one that is nonetheless everywhere smooth and defined.
    return np.where(np.isnan(wmean_dist), dist.mean(), wmean_dist)


def discrete_genotype_dissimilarity(x, y, q=2):
    "Dissimilarity between 1D genotypes, accounting for fuzzyness."
    x = genotype_binary_to_sign(x)
    y = genotype_binary_to_sign(y)
    x_sign, y_sign = np.sign(x), np.sign(y)

    dist = np.abs((x_sign - y_sign) / 2)
    weight = np.abs(x * y)
    wmean_dist = ((weight * dist).sum()) / ((weight.sum()))
    # Why not finish up by powering it by (1 / q)?
    # I don't do this part because it loses the city-block distance
    # interpretation when x and y are both discrete (i.e. one of {0, 1}).

    # While the basic function is undefined where weight.sum() == 0
    # (and this is only true when one of x or y is always exactly 0.5 at every
    # index),
    # the limit approaches the same value from both directions.
    # We therefore redefine the dissimilarity as a piecewise function,
    # but one that is nonetheless everywhere smooth and defined.
    return np.where(np.isnan(wmean_dist), dist.mean(), wmean_dist)


def genotype_masked_hamming_distance(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    diff = np.abs(x - y)
    return diff[mask].sum() / np.ones_like(diff)[mask].sum()


def genotype_cdist(xx, yy, q=2):
    return cdist(xx, yy, genotype_dissimilarity, q=q)


def discrete_genotype_cdist(xx, yy):
    return cdist(xx, yy, discrete_genotype_dissimilarity)


def genotype_masked_hamming_cdist(xx, yy):
    return cdist(xx, yy, genotype_masked_hamming_distance)


def genotype_pdist(xx, quiet=True, q=2):
    if not quiet:
        warnings.warn("Progress bar not implemented for genotype_pdist.")
    return pdist(xx, genotype_dissimilarity, q=q)


def adjusted_community_dissimilarity(x, y, gdiss):
    gdiss = np.asarray(gdiss)
    outer = np.einsum("a,b->ab", x, y)
    expect = outer * gdiss
    return expect.sum()
