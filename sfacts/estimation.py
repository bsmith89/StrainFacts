
from sklearn.cluster import AgglomerativeClustering
from sfacts.genotype import genotype_pdist, mask_missing_genotype
from sfacts.pyro_util import all_torch

import pandas as pd
import numpy as np
import scipy as sp
from scipy.spatial.distance import squareform

import pyro
import pyro.distributions as dist
import torch

from tqdm import tqdm
from sfacts.logging_util import info

from sfacts.metagenotype_model import condition_model

def cluster_genotypes(
    gamma, thresh, progress=False, precomputed_pdist=None
):
    
    if precomputed_pdist is None:
        compressed_dmat = genotype_pdist(gamma, progress=progress)
    else:
        compressed_dmat = precomputed_pdist

    clust = pd.Series(
        AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=thresh,
        )
        .fit(squareform(compressed_dmat))
        .labels_
    )

    return clust, compressed_dmat

def initialize_parameters_by_clustering_samples(
    y, m, thresh, additional_strains_factor=0.5, progress=False, precomputed_pdist=None,
):
    n, g = y.shape

    sample_genotype = (y + 1) / (m + 2)
    clust, cdmat = cluster_genotypes(sample_genotype, thresh=thresh, progress=progress, precomputed_pdist=precomputed_pdist)

    y_total = (
        pd.DataFrame(pd.DataFrame(y))
        .groupby(clust)
        .sum()
        .values
    )
    m_total = (
        pd.DataFrame(pd.DataFrame(m))
        .groupby(clust)
        .sum()
        .values
    )
    clust_genotype = (y_total + 1) / (m_total + 2)
    additional_haplotypes = int(
        additional_strains_factor * clust_genotype.shape[0]
    )


    gamma_init = pd.concat(
        [
            pd.DataFrame(clust_genotype),
            pd.DataFrame(np.ones((additional_haplotypes, g)) * 0.5),
        ]
    ).values

    s_init = gamma_init.shape[0]
    pi_init = np.ones((n, s_init))
    for i in range(n):
        pi_init[i, clust[i]] = s_init - 1
    pi_init /= pi_init.sum(1, keepdims=True)

    assert (~np.isnan(gamma_init)).all()

    return gamma_init, pi_init, cdmat


def estimate_parameters(
    model,
    data,
    dtype=torch.float32,
    device='cpu',
    initialize_params=None,
    maxiter=10000,
    lag=100,
    lr=1e-0,
    clip_norm=100,
    progress=True,
    **model_kwargs,
):
    conditioned_model = condition_model(
        model,
        data=data,
        dtype=dtype,
        device=device,
        **model_kwargs,
    )
    if initialize_params is None:
        initialize_params = {}

    _guide = pyro.infer.autoguide.AutoLaplaceApproximation(
        conditioned_model,
        init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(
            values=all_torch(**initialize_params, dtype=dtype, device=device)
        ),
    )
    opt = pyro.optim.Adamax({"lr": lr}, {"clip_norm": clip_norm})
    svi = pyro.infer.SVI(
        conditioned_model,
        _guide,
        opt,
        loss=pyro.infer.JitTrace_ELBO()
    )
    pyro.clear_param_store()

    history = []
    pbar = tqdm(range(maxiter), disable=(not progress))
    try:
        for i in pbar:
            elbo = svi.step()

            if np.isnan(elbo):
                raise RuntimeError("ELBO NaN?")

            # Fit tracking
            history.append(elbo)

            # Reporting/Breaking
            if (i % 10 == 0):
                if i > lag:
                    delta = history[-2] - history[-1]
                    delta_lag = (history[-lag] - history[-1]) / lag
                    if delta_lag <= 0:
                        if progress:
                            info("Converged")
                        break
                    pbar.set_postfix({
                        'ELBO': history[-1],
                        'delta': delta,
                        f'lag{lag}': delta_lag,
                    })
    except KeyboardInterrupt:
        info("Interrupted")
        pass
    est = pyro.infer.Predictive(conditioned_model, guide=_guide, num_samples=1)()
    est = {
        k: est[k].detach().cpu().numpy().mean(0).squeeze()
        for k in est.keys()
    }
    return est, history


def merge_similar_genotypes(
    gamma, pi, thresh, delta=None, progress=False,
):
    if delta is None:
        delta = np.ones_like(gamma)
        
    gamma_adjust = mask_missing_genotype(gamma, delta)

    clust, dmat = cluster_genotypes(gamma_adjust, thresh=thresh, progress=progress)
    gamma_mean = (
        pd.DataFrame(pd.DataFrame(gamma_adjust))
        .groupby(clust)
        .apply(lambda x: sp.special.expit(sp.special.logit(x)).mean(0))
        .values
    )
    delta_mean = (
        pd.DataFrame(pd.DataFrame(delta))
        .groupby(clust)
        .mean()
        .values
    )
    pi_sum = (
        pd.DataFrame(pd.DataFrame(pi))
        .groupby(clust, axis='columns')
        .sum()
        .values
    )
    
    return gamma_mean, pi_sum, delta_mean
