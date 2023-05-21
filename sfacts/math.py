import numpy as np
from scipy.spatial.distance import cdist, pdist


def minor_frequency(p):
    return -(np.abs(p * 2 - 1) - 1) / 2


def binary_entropy(p):
    q = 1 - p
    return np.nan_to_num(-(p * np.log2(p) + q * np.log2(q)))


def entropy(p, axis=-1):
    return np.nan_to_num(-(p * np.log2(p)).sum(axis))


def genotype_binary_to_sign(p):
    return 2 * p - 1


def genotype_dissimilarity(x, y, q=2):
    "Dissimilarity between 1D genotypes, accounting for fuzzyness."
    assert q == 2, "Not implemented"
    x = genotype_binary_to_sign(x)
    y = genotype_binary_to_sign(y)

    # FIXME: This sqrt(abs(.)) is a no-op
    dist = np.abs(x - y)
    weight = np.sqrt(np.abs(x * y))
    wmean_dist = (weight * dist).sum() / weight.sum()

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


def genotype_pdist(xx, q=2):
    return pdist(xx, genotype_dissimilarity, q=q)


def adjusted_community_dissimilarity(x, y, gdiss):
    gdiss = np.asarray(gdiss)
    outer = np.einsum("a,b->ab", x, y)
    expect = outer * gdiss
    return expect.sum()


def podlesny_dissimilarity(mx, my):
    mx = mx > 0
    my = my > 0
    count_shared = (((mx & my).any(1))).sum()
    count_covered = (mx.any(1) & my.any(1)).sum()
    return 1 - count_shared / count_covered


def podlesny_cdist(xx, yy):
    xx = xx > 0
    yy = yy > 0
    count_shared = (np.einsum("aij,bij->abi", xx.astype(int), yy.astype(int)) > 0).sum(
        -1
    )
    count_covered = np.einsum("ai,bi->ab", xx.any(2).astype(int), yy.any(2).astype(int))
    return 1 - count_shared / count_covered
