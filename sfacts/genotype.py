
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy as sp
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm

def prob_to_sign(gamma):
    return gamma * 2 - 1

def genotype_distance(x, y):
    x = prob_to_sign(x)
    y = prob_to_sign(y)
    dist = ((x - y) / 2) ** 2
    weight = (x * y) ** 2
    wmean_dist = ((weight * dist).mean()) / ((weight.mean()))
    return wmean_dist

def sign_genotype_distance(x, y):
    dist = ((x - y) / 2) ** 2
    weight = (x * y) ** 2
    wmean_dist = ((weight * dist).mean()) / ((weight.mean()))
    return wmean_dist

def genotype_pdist(gamma, progress=False):
    metric = sign_genotype_distance
    X = np.asarray(gamma * 2 - 1)
    m = X.shape[0]
    dm = np.empty((m * (m - 1)) // 2)
    k = 0
    with tqdm(total=len(dm), disable=(not progress)) as pbar:
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                dm[k] = metric(X[i], X[j])
                k = k + 1
                pbar.update()
    return dm
    

def counts_to_p_estimate(y, m, pseudo=1):
    return (y + pseudo) / (m + pseudo * 2)

def genotype_linkage(gamma, progress=False, **kwargs):
    dmat = genotype_pdist(gamma, progress=progress)
    kw = dict(method='complete')
    kw.update(kwargs)
    return linkage(dmat, **kw), dmat

def mask_missing_genotype(gamma, delta):
    return sp.special.expit(sp.special.logit(gamma) * delta)
