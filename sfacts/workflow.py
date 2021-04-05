
import pyro
from sfacts.pandas_util import idxwhere
from sfacts.metagenotype_model import model, simulate, condition_model
from sfacts.estimation import (
    initialize_parameters_by_clustering_samples,
    estimate_parameters,
    merge_similar_genotypes
)
from sfacts.genotype import mask_missing_genotype
from sfacts.evaluation import match_genotypes, sample_mean_masked_genotype_entropy, community_accuracy_test
from sfacts.data import load_input_data, select_informative_positions
import time
import numpy as np
from sfacts.logging_util import info

def fit_to_data(
    y,
    m,
    fit_kwargs,
    preclust=True,
    preclust_kwargs=None,
    postclust=True,
    postclust_kwargs=None,
    seed=1,
    quiet=False
):
    n, g = y.shape
    info(f"Setting RNG seed to {seed}.", quiet=quiet)
    pyro.util.set_rng_seed(seed)
    if preclust:
        info(f"Preclustering {n} samples based on {g} positions.", quiet=quiet)
        assert preclust_kwargs is not None
        gamma_init, pi_init, cdmat = initialize_parameters_by_clustering_samples(
            y,
            m,
            **preclust_kwargs
        )
        initialize_params=dict(gamma=gamma_init, pi=pi_init)
        s_fit = gamma_init.shape[0]
        info(f"Estimated {s_fit} strains from {n} samples.", quiet=quiet)
    else:
        initialize_params = None
        s_fit = fit_kwargs.pop('s')

    info(f"Optimizing model parameters.", quiet=quiet)
    fit, history = estimate_parameters(
        model,
        data=dict(y=y, m=m),
        n=n,
        g=g,
        s=s_fit,
        initialize_params=initialize_params,
        **fit_kwargs,
    )
    
    if postclust:
        info(f"Dereplicating highly similar strains.", quiet=quiet)
        merge_gamma, merge_pi, merge_delta = merge_similar_genotypes(
            fit['gamma'],
            fit['pi'],
            delta=fit['delta'],
            **postclust_kwargs,
        )
        mrg = fit.copy()
        mrg['gamma'] = merge_gamma
        mrg['pi'] = merge_pi
        mrg['delta'] = merge_delta
        s_mrg = mrg['gamma'].shape[0]
        info(f"Original {s_fit} strains down to {s_mrg} after dereplication.", quiet=quiet)
    else:
        mrg = fit
        
    return mrg, fit, history

def simulate_fit_and_evaluate(
    s_sim,
    n_sim,
    g_sim,
    n_fit,
    g_fit,
    sim_kwargs,
    fit_kwargs,
    seed_sim=1,
    seed_fit=1,
    preclust=True,
    preclust_kwargs=None,
    postclust=True,
    postclust_kwargs=None,
    quiet=False,
):
    info(f"Setting RNG seed to {seed_sim}.", quiet=quiet)
    pyro.util.set_rng_seed(seed_sim)
    info(f"Simulating data from model.", quiet=quiet)
    sim = simulate(
        condition_model(
            model,
            n=n_sim,
            g=g_sim,
            s=s_sim,
            **sim_kwargs,
        )
    )
    
    info(f"Starting runtime clock.", quiet=quiet)
    start_time = time.time()
    mrg, fit, history = fit_to_data(
        sim['y'][:n_fit, :g_fit],
        sim['m'][:n_fit, :g_fit],
        fit_kwargs=fit_kwargs,
        preclust=preclust,
        preclust_kwargs=preclust_kwargs,
        postclust=postclust,
        postclust_kwargs=postclust_kwargs,
        seed=seed_fit,
        quiet=quiet,
    )
    end_time = time.time()
    info(f"Stopping runtime clock.", quiet=quiet)
    
    info(f"Calculating statistics.", quiet=quiet)
    s_mrg = mrg['gamma'].shape[0]
    
    sim_gamma_adj = mask_missing_genotype(sim['gamma'][:, :g_fit], sim['delta'][:, :g_fit])
    mrg_gamma_adj = mask_missing_genotype(mrg['gamma'], mrg['delta'])
    best_hit, best_dist = match_genotypes(sim_gamma_adj, mrg_gamma_adj)
    weighted_mean_genotype_error = (best_dist * sim['pi'][:n_fit].mean(0)).sum()
    runtime = end_time - start_time
    
    _, _, beta_diversity_error_ratio, _ = (
        community_accuracy_test(sim['pi'][:n_fit], mrg['pi'])
    )
    
    strain_count_error = s_mrg - s_sim
    
    mean_sample_weighted_genotype_entropy = (
        sample_mean_masked_genotype_entropy(mrg['pi'], mrg['gamma'], mrg['delta']).mean()
    )
    
    return (
        weighted_mean_genotype_error,
        beta_diversity_error_ratio,
        strain_count_error,
        mean_sample_weighted_genotype_entropy,
        runtime,
        sim,
        mrg
    )


def filter_data(
    data, 
    incid_thresh=0.1,
    cvrg_thresh=0.15,
):
    info("Filtering positions.")
    informative_positions = select_informative_positions(
        data, incid_thresh
    )
    npos_available = len(informative_positions)
    info(
        f"Found {npos_available} informative positions with minor "
        f"allele incidence of >{incid_thresh}"
    )

    info("Filtering libraries.")
    suff_cvrg_samples = idxwhere(
        (
            (
                data.sel(position=informative_positions).sum(["allele"]) > 0
            ).mean("position")
            > cvrg_thresh
        ).to_series()
    )
    nlibs = len(suff_cvrg_samples)
    info(
        f"Found {nlibs} libraries with >{cvrg_thresh:0.1%} "
        f"of informative positions covered."
    )
    return informative_positions, suff_cvrg_samples

def sample_positions(
    informative_positions,
    npos=1000,
    seed=None,
):
    if seed is not None:
        info(f"Setting RNG seed to {seed}.")
        np.random.seed(seed)
    npos_available = len(informative_positions)
    _npos = min(npos, npos_available)
    info(f"Randomly sampling {npos} positions.")
    position_ss = np.random.choice(
        informative_positions,
        size=_npos,
        replace=False,
    )
    return position_ss  

def filter_subsample_and_fit(
    data,
    incid_thresh=0.1,
    cvrg_thresh=0.15,
    npos=1000,
    seed=1,
    **fit_to_data_kwargs
):
    info(f"Full data shape: {data.sizes}.")
    informative_positions, suff_cvrg_samples = filter_data(
        data, incid_thresh=incid_thresh, cvrg_thresh=cvrg_thresh
    )
    position_ss = sample_positions(informative_positions, npos, seed=seed)
    info("Constructing input data.")
    data_fit = data.sel(library_id=suff_cvrg_samples, position=position_ss)
    m_ss = data_fit.sum("allele")
    n, g_ss = m_ss.shape
    y_obs_ss = data_fit.sel(allele="alt")  
    mrg_ss, fit_ss, history = fit_to_data(
        y_obs_ss.values,
        m_ss.values,
        seed=seed,
        **fit_to_data_kwargs,
    )
    return mrg_ss, data_fit, history
