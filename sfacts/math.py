import numpy as np
from scipy.spatial.distance import cdist, pdist
import warnings


def binary_entropy(p):
    p = np.clip(p, np.nextafter(0, 1), np.nextafter(1, 0))
    q = 1 - p
    return -(p * np.log2(p) + q * np.log2(q))


def entropy(p, axis=-1):
    p = np.clip(p, np.nextafter(0, 1), np.nextafter(1, 0))
    return -(p * np.log2(p)).sum(axis)


def genotype_binary_to_sign(p):
    return 2 * p - 1


def genotype_dissimilarity(x, y):
    "Dissimilarity between 1D genotypes, accounting for fuzzyness."
    x = genotype_binary_to_sign(x)
    y = genotype_binary_to_sign(y)

    dist = ((x - y) / 2) ** 2
    weight = np.abs(x * y)
    wmean_dist = ((weight * dist).sum()) / ((weight.sum()))

    # While the basic function is undefined where weight.sum() == 0
    # (and this is only true when one of x or y is always exactly 0.5 at every
    # index),
    # the limit approaches the same value from both directions.
    # We therefore redefine the dissimilarity as a piecewise function,
    # but one that is nonetheless everywhere smooth and defined.
    return np.where(np.isnan(wmean_dist), dist.mean(), wmean_dist)


def genotype_cdist(xx, yy):
    return cdist(xx, yy, genotype_dissimilarity)


def genotype_pdist(xx, quiet=True):
    if not quiet:
        warnings.warn("Progress bar not implemented for genotype_pdist.")
    return pdist(xx, genotype_dissimilarity)


def adjusted_community_dissimilarity(x, y, gdiss):
    gdiss = np.asarray(gdiss)
    outer = np.einsum("a,b->ab", x, y)
    expect = outer * gdiss
    return expect.sum()
