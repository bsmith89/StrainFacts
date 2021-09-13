import numpy as np
from scipy.spatial.distance import cdist, pdist
import warnings


def binary_entropy(p):
    q = 1 - p
    return -(p * np.log2(p) + q * np.log2(q))


def entropy(p, axis=-1):
    return -(p * np.log2(p)).sum(axis)


def genotype_binary_to_sign(p):
    return 2 * p - 1


def genotype_dissimilarity(x, y, pseudo=0.0):
    "Dissimilarity between 1D genotypes, accounting for fuzzyness."
    x = genotype_binary_to_sign(x)
    y = genotype_binary_to_sign(y)

    dist = ((x - y) / 2) ** 2
    weight = np.abs(x * y)
    wmean_dist = ((weight * dist).sum() + pseudo) / ((weight.sum() + pseudo))
    return wmean_dist


def genotype_cdist(xx, yy, pseudo=0.0):
    return cdist(xx, yy, genotype_dissimilarity, pseudo=pseudo)


def genotype_pdist(xx, pseudo=0.0, quiet=True):
    if not quiet:
        warnings.warn("Progress bar not implemented for genotype_pdist.")
    return pdist(xx, genotype_dissimilarity, pseudo=pseudo)
