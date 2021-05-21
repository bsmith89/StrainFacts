import sfacts as sf
import pyro
import time


simple_fit = sf.estimation.estimate_parameters


def simulation_benchmark(
    nsample,
    nposition,
    sim_nstrain,
    fit_nstrain,
    sim_model,
    fit_model=None,
    sim_data=None,
    sim_hyperparameters=None,
    sim_seed=None,
    fit_data=None,
    fit_hyperparameters=None,
    fit_seed=None,
    opt=pyro.optim.Adamax({"lr": 1e-0}, {"clip_norm": 100}),
    **fit_kwargs,
):
    if fit_model is None:
        fit_model = sim_model

    sim = sf.model.ParameterizedModel(
        sim_model,
        coords=dict(
            sample=nsample,
            position=nposition,
            allele=["alt", "ref"],
            strain=sim_nstrain,
        ),
        data=sim_data,
        hyperparameters=sim_hyperparameters,
    ).simulate_world(seed=sim_seed)

    start_time = time.time()
    est, history, *_ = simple_fit(
        sf.model.ParameterizedModel(
            fit_model,
            coords=dict(
                sample=nsample,
                position=nposition,
                allele=["alt", "ref"],
                strain=fit_nstrain,
            ),
            data=fit_data,
            hyperparameters=fit_hyperparameters,
        ).condition(
            **sim.metagenotypes.to_counts_and_totals(),
        ),
        opt=opt,
        seed=fit_seed,
        **fit_kwargs,
    )
    end_time = time.time()

    return (
        sf.evaluation.weighted_genotype_error(sim, est),
        sf.evaluation.community_error(sim, est),
        end_time - start_time,
    )


def fit_fix_pi_refit(
    model,
    stage2_hyperparameters,
    initialize_params=None,
    **kwargs,
):
    est0, history0, *_ = simple_fit(
        model, initialize_params=initialize_params, **kwargs
    )
    est1, history1, *_ = simple_fit(
        model.with_hyperparameters(**stage2_hyperparameters).condition(
            pi=est0.communities.values
        ),
        **kwargs,
    )
    return est1, history0 + history1, est0


def fit_fix_pi_refit_merge_strains_refit(
    model,
    thresh,
    initialize_params=None,
    stage2_hyperparameters=None,
    **kwargs,
):
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}
    est1, history1, est0, *_ = fit_fix_pi_refit(
        model,
        stage2_hyperparameters,
        initialize_params=initialize_params,
        **kwargs,
    )
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est1, thresh=thresh
    )
    est2, history2 = simple_fit(
        model.with_amended_coords(strain=agg_communities.strain)
        .with_hyperparameters(**stage2_hyperparameters)
        .condition(pi=agg_communities.values),
        **kwargs,
    )
    return est2, history1 + history2, est0, est1


def fit_fix_pi_refit_merge_strains_unfix_pi_refit(
    model,
    thresh,
    initialize_params=None,
    stage2_hyperparameters=None,
    **kwargs,
):
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}
    est1, history1, est0, *_ = fit_fix_pi_refit(
        model,
        stage2_hyperparameters,
        initialize_params=initialize_params,
        **kwargs,
    )
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est1, thresh=thresh
    )
    est2, history2 = simple_fit(
        model.with_amended_coords(strain=agg_communities.strain),
        **kwargs,
    )
    return est2, history1 + history2, est0, est1
