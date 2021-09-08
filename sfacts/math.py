import numpy as np


def binary_entropy(p):
    q = 1 - p
    return -(p * np.log2(p) + q * np.log2(q))


def entropy(p, axis=-1):
    return -(p * np.log2(p)).sum(axis)
