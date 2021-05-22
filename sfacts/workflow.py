import sfacts as sf
import pyro
import time


fit_simple = sf.estimation.estimate_parameters


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
    est, history, *_ = fit_simple(
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


def fit_then_relax_genotypes(
    model,
    stage2_hyperparameters,
    initialize_params=None,
    **kwargs,
):
    est0, history0 = fit_simple(model, initialize_params=initialize_params, **kwargs)
    est1, history1 = fit_simple(
        model.with_hyperparameters(**stage2_hyperparameters).condition(
            # FIXME: Drop conditining on delta for consistency with 3-stage workflow?
            delta=est0.data.missingness.values,
            pi=est0.data.communities.values,
            mu=est0.data.mu.values,
            alpha=est0.data.alpha.values,
            epsilon=est0.data.epsilon.values,
            m_hyper_r=est0.data.m_hyper_r.values,
        ),
        **kwargs,
    )
    return (est0, est1), (history0, history1)


def fit_then_relax_genotypes_and_collapse(
    model,
    thresh,
    initialize_params=None,
    stage2_hyperparameters=None,
    **kwargs,
):
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}

    est0, history0 = fit_simple(
        model,
        initialize_params=initialize_params,
        **kwargs,
    )
    est1, history1 = fit_simple(
        model.with_hyperparameters(**stage2_hyperparameters).condition(
            pi=est0.data.communities.values,
            mu=est0.data.mu.values,
            alpha=est0.data.alpha.values,
            epsilon=est0.data.epsilon.values,
            m_hyper_r=est0.data.m_hyper_r.values,
        ),
        **kwargs,
    )
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est1, thresh=thresh
    )
    est2, history2 = fit_simple(
        model.with_amended_coords(strain=agg_communities.strain).condition(
            pi=agg_communities.values,
            mu=est1.data.mu.values,
            alpha=est1.data.alpha.values,
            epsilon=est1.data.epsilon.values,
            m_hyper_r=est1.data.m_hyper_r.values,
        ),
        **kwargs,
    )
    est3, history3 = fit_simple(
        model.with_amended_coords(strain=agg_communities.strain)
        .with_hyperparameters(**stage2_hyperparameters)
        .condition(
            delta=est2.missingness.values,
            pi=est2.communities.values,
            mu=est2.data.mu.values,
            alpha=est2.data.alpha.values,
            epsilon=est2.data.epsilon.values,
            m_hyper_r=est2.data.m_hyper_r.values,
        ),
        **kwargs,
    )
    return (est0, est1, est2, est3), (history0, history1, history2, history3)
