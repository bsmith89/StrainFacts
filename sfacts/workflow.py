
import pyro
from sfacts.metagenotype_model import model, simulate, condition_model
from sfacts.estimation import (
    initialize_parameters_by_clustering_samples,
    estimate_parameters,
    merge_similar_genotypes
)
from sfacts.genotype import mask_missing_genotype
from sfacts.evaluation import match_genotypes, sample_mean_masked_genotype_entropy, community_accuracy_test
import time

def fit_to_data(
    y,
    m,
    fit_kwargs,
    preclust=True,
    preclust_kwargs=None,
    postclust=True,
    postclust_kwargs=None,
    seed=1,
):
    n, g = y.shape
    pyro.util.set_rng_seed(seed)
    if preclust:
        assert preclust_kwargs is not None
        gamma_init, pi_init = initialize_parameters_by_clustering_samples(
            y,
            m,
            **preclust_kwargs
        )
        initialize_params=dict(gamma=gamma_init, pi=pi_init)
        s = gamma_init.shape[0]
    else:
        initialize_params = None
        s = fit_kwargs.pop('s')

    fit, history = estimate_parameters(
        model,
        data=dict(y=y, m=m),
        n=n,
        g=g,
        s=s,
        initialize_params=initialize_params,
        **fit_kwargs,
    )
    
    if postclust:
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
    seed=1,
    preclust=True,
    preclust_kwargs=None,
    postclust=True,
    postclust_kwargs=None,
    return_raw=False,
):
    pyro.util.set_rng_seed(seed)
    sim = simulate(
        condition_model(
            model,
            n=n_sim,
            g=g_sim,
            s=s_sim,
            **sim_kwargs,
        )
    )
    
    start_time = time.time()
    mrg, fit, history = fit_to_data(
        sim['y'][:n_fit, :g_fit],
        sim['m'][:n_fit, :g_fit],
        fit_kwargs=fit_kwargs,
        preclust=preclust,
        preclust_kwargs=preclust_kwargs,
        postclust=postclust,
        postclust_kwargs=postclust_kwargs,
        seed=seed,
    )
    end_time = time.time()
    
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
