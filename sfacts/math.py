import numpy as np


def binary_entropy(p):
    q = 1 - p
    return -(p * np.log2(p) + q * np.log2(q))
