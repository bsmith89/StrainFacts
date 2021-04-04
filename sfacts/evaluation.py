from scipy.spatial.distance import cdist, pdist
import pandas as pd
import numpy as np


def binary_entropy(p):
    q = 1 - p
    ent = -(p * np.log2(p) + q * np.log2(q))
    return ent

def sum_binary_entropy(p, normalize=False, axis=None):
    q = 1 - p
    ent = np.sum(-(p * np.log2(p) + q * np.log2(q)), axis=axis)
    if normalize:
        ent = ent / p.shape[axis]
    return ent

def mean_masked_genotype_entropy(gamma, delta):
    return (binary_entropy(gamma) * delta).mean(1)

def sample_mean_masked_genotype_entropy(pi, gamma, delta):
    return (pi @ mean_masked_genotype_entropy(gamma, delta).reshape((-1, 1))).squeeze()

def match_genotypes(gammaA, gammaB):
    g = gammaA.shape[1]
    dist = pd.DataFrame(cdist(gammaA, gammaB, metric='cityblock'))
    return dist.idxmin(axis=1), dist.min(axis=1) / g

def _rmse(x, y):
    return np.sqrt(np.square(x - y).mean())

def community_accuracy_test(pi_sim, pi_fit, reps=99):
    bc_sim = 1 - pdist(pi_sim, metric='braycurtis')
    bc_fit = 1 - pdist(pi_fit, metric='braycurtis')
    err = _rmse(bc_sim, bc_fit)
    
    null = []
    n = len(bc_sim)
    for i in range(reps):
        bc_sim_permute = np.random.permutation(bc_sim)
        null.append(_rmse(bc_sim, bc_sim_permute))
    null = np.array(null)
    
    return err, null, err / np.mean(null), (np.sort(null) < err).mean()
