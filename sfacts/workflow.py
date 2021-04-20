import pyro
from sfacts.pandas_util import idxwhere
from sfacts.model import model, simulate, condition_model
from sfacts.estimation import (
    initialize_parameters_by_clustering_samples,
    initialize_parameters_by_nmf,
    estimate_parameters,
    merge_similar_genotypes,
)
from sfacts.genotype import mask_missing_genotype
from sfacts.evaluation import (
    match_genotypes,
    sample_mean_masked_genotype_entropy,
    community_accuracy_test,
    metacommunity_composition_rss,
)
from sfacts.data import load_input_data, select_informative_positions
import time
import numpy as np
from sfacts.logging_util import info


def fit_to_data(
    y,
    m,
    fit_kwargs,
    initialize="nmf",
    initialize_kwargs=None,
    postclust=True,
    postclust_kwargs=None,
    seed=1,
    quiet=False,
    additional_conditioning_data=None,
):
    if additional_conditioning_data is None:
        additional_conditioning_data = {}

    n, g = y.shape
    info(f"Setting RNG seed to {seed}.", quiet=quiet)
    pyro.util.set_rng_seed(seed)
    if initialize == "nmf":
        info(f"Initializing {n} samples and {g} positions using NMF.", quiet=quiet)
        assert initialize_kwargs is not None
        gamma_init, pi_init, _ = initialize_parameters_by_nmf(
            y, m, random_state=seed, **initialize_kwargs
        )
        initialize_params = dict(gamma=gamma_init, pi=pi_init)
        s_fit = gamma_init.shape[0]
        info(f"Initialized {s_fit} strains in {n} samples.", quiet=quiet)
    elif initialize == "clust":
        info(
            f"Initializing {n} samples and {g} positions using hierarchical clustering.",
            quiet=quiet,
        )
        assert initialize_kwargs is not None
        gamma_init, pi_init, _ = initialize_parameters_by_clustering_samples(
            y, m, **initialize_kwargs
        )
        initialize_params = dict(gamma=gamma_init, pi=pi_init)
        s_fit = gamma_init.shape[0]
        info(f"Initialized {s_fit} strains in {n} samples.", quiet=quiet)
    elif not initialize:
        initialize_params = None
        s_fit = fit_kwargs.pop("s")
    else:
        raise NotImplementedError(f"Initializing strategy: '{initialize}' not known.")

    info(f"Optimizing model parameters.", quiet=quiet)
    info(f"Setting RNG seed to {seed}.", quiet=quiet)
    pyro.util.set_rng_seed(seed)
    fit, history = estimate_parameters(
        model,
        data=dict(y=y, m=m, **additional_conditioning_data),
        n=n,
        g=g,
        s=s_fit,
        initialize_params=initialize_params,
        **fit_kwargs,
    )

    if postclust:
        info(f"Dereplicating highly similar strains.", quiet=quiet)
        merge_gamma, merge_pi, merge_delta = merge_similar_genotypes(
            fit["gamma"],
            fit["pi"],
            delta=fit["delta"],
            **postclust_kwargs,
        )
        mrg = fit.copy()
        mrg["gamma"] = merge_gamma
        mrg["pi"] = merge_pi
        mrg["delta"] = merge_delta
        s_mrg = mrg["gamma"].shape[0]
        info(
            f"Original {s_fit} strains down to {s_mrg} after dereplication.",
            quiet=quiet,
        )
    else:
        mrg = fit
    info(f"Finished fitting to data.", quiet=quiet)

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
        sim["y"][:n_fit, :g_fit],
        sim["m"][:n_fit, :g_fit],
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
    s_mrg = mrg["gamma"].shape[0]

    sim_gamma_adj = mask_missing_genotype(
        sim["gamma"][:, :g_fit], sim["delta"][:, :g_fit]
    )
    mrg_gamma_adj = mask_missing_genotype(mrg["gamma"], mrg["delta"])
    best_hit, best_dist = match_genotypes(sim_gamma_adj, mrg_gamma_adj)
    weighted_mean_genotype_error = (best_dist * sim["pi"][:n_fit].mean(0)).sum()
    runtime = end_time - start_time

    _, _, beta_diversity_error_ratio, _ = community_accuracy_test(
        sim["pi"][:n_fit], mrg["pi"]
    )

    metacommunity_composition_error = metacommunity_composition_rss(
        sim["pi"], mrg["pi"]
    )

    mean_sample_weighted_genotype_entropy = sample_mean_masked_genotype_entropy(
        mrg["pi"], mrg["gamma"], mrg["delta"]
    ).mean()
    info(f"Finished calculating statistics.", quiet=quiet)

    return (
        weighted_mean_genotype_error,
        beta_diversity_error_ratio,
        metacommunity_composition_error,
        mean_sample_weighted_genotype_entropy,
        runtime,
        sim,
        mrg,
    )


def filter_data(
    data,
    incid_thresh=0.1,
    cvrg_thresh=0.15,
):
    info("Filtering positions.")
    informative_positions = select_informative_positions(data, incid_thresh)
    npos_available = len(informative_positions)
    info(
        f"Found {npos_available} informative positions with minor "
        f"allele incidence of >{incid_thresh}"
    )

    info("Filtering libraries.")
    suff_cvrg_samples = idxwhere(
        (
            (data.sel(position=informative_positions).sum(["allele"]) > 0).mean(
                "position"
            )
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
    info(f"Finished sampling.")
    return position_ss


def filter_subsample_and_fit(
    data, incid_thresh=0.1, cvrg_thresh=0.15, npos=1000, seed=1, **fit_to_data_kwargs
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


def filter_subsample_fit_and_refit_genotypes(
    data, fit_kwargs, incid_thresh=0.1, cvrg_thresh=0.15, npos=1000, seed=1, **kwargs
):
    info(f"Full data shape: {data.sizes}.")
    informative_positions, suff_cvrg_samples = filter_data(
        data, incid_thresh=incid_thresh, cvrg_thresh=cvrg_thresh
    )
    position_ss = sample_positions(informative_positions, npos, seed=seed)
    info("Constructing input data.")
    data_filt = data.sel(library_id=suff_cvrg_samples)
    data_ss = data_filt.sel(position=position_ss)
    m_ss = data_ss.sum("allele")
    n, g_ss = m_ss.shape
    y_obs_ss = data_ss.sel(allele="alt")
    mrg_ss, fit_ss, history = fit_to_data(
        y_obs_ss.values,
        m_ss.values,
        seed=seed,
        fit_kwargs=fit_kwargs,
        **kwargs,
    )

    info(f"Refitting genotypes at all positions")
    s = mrg_ss["gamma"].shape[0]
    refit_kwargs = fit_kwargs.copy()
    if s in refit_kwargs:
        del refit_kwargs["s"]
    n = len(suff_cvrg_samples)
    g_total = len(informative_positions)
    fixed = mrg_ss.copy()
    for k in ["gamma", "delta", "nu", "m", "p_noerr", "p", "y", "rho"]:
        del fixed[k]

    y = data_filt.sel(allele="alt").values
    m = data_filt.sum("allele").values
    out = fixed.copy()
    gamma_out = []
    delta_out = []
    nu_out = []
    m_out = []
    p_noerr_out = []
    p_out = []
    y_out = []
    nwindows = g_total // npos + 1
    for window_i, j_start in enumerate(range(0, g_total, npos)):
        window_ip1 = window_i + 1
        info(f"Fitting genotype window {window_ip1} of {nwindows}.")
        j_stop = min(j_start + g_ss, g_total)
        refit, history = estimate_parameters(
            model,
            data=dict(y=y[:, j_start:j_stop], m=m[:, j_start:j_stop], **fixed),
            n=n,
            g=j_stop - j_start,
            s=s,
            **refit_kwargs,
        )
        gamma_out.append(refit["gamma"])
        delta_out.append(refit["delta"])
        nu_out.append(refit["nu"])
        m_out.append(refit["m"])
        p_noerr_out.append(refit["p_noerr"])
        p_out.append(refit["p"])
        y_out.append(refit["y"])
        info(f"Finished fitting genotype window {window_ip1} of {nwindows}.")
    info(f"Finished all windows.")

    out["gamma"] = np.concatenate(gamma_out, axis=1)
    out["delta"] = np.concatenate(delta_out, axis=1)
    out["nu"] = np.concatenate(nu_out, axis=1)
    out["m"] = np.concatenate(m_out, axis=1)
    out["p_noerr"] = np.concatenate(p_noerr_out, axis=1)
    out["p"] = np.concatenate(p_out, axis=1)
    out["y"] = np.concatenate(y_out, axis=1)
    info(f"Finished constructing arrays.")

    return out, data_filt, informative_positions, position_ss
