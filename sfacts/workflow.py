import sfacts as sf
import pyro
import time
import torch


def _chunk_start_end_iterator(total, per):
    for i in range(total // per):
        yield (per * i), (per * (i + 1))
    if (i + 1) * per < total:
        yield (i + 1) * per, total


def fit_metagenotypes_simple(
    structure,
    metagenotypes,
    nstrain,
    hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
):
    if estimation_kwargs is None:
        estimation_kwargs = {}

    sf.logging_util.info(
        f"START: Fitting data with shape {metagenotypes.sizes}.", quiet=quiet
    )
    model = sf.model.ParameterizedModel(
        structure,
        coords=dict(
            sample=metagenotypes.sample.values,
            position=metagenotypes.position.values,
            allele=metagenotypes.allele.values,
            strain=range(nstrain),
        ),
        hyperparameters=hyperparameters,
        data=condition_on,
        device=device,
        dtype=dtype,
    ).condition(**metagenotypes.to_counts_and_totals())
    start_time = time.time()
    est, history = sf.estimation.estimate_parameters(
        model,
        quiet=quiet,
        **estimation_kwargs,
    )
    end_time = time.time()
    delta_time = end_time - start_time
    sf.logging_util.info(f"END: Fit in {delta_time} seconds.", quiet=quiet)
    return est, history


def fit_metagenotypes_then_refit_genotypes(
    structure,
    metagenotypes,
    nstrain,
    hyperparameters=None,
    stage2_hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
):
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}

    _estimate_parameters = lambda model: sf.estimation.estimate_parameters(
        model, quiet=quiet, **estimation_kwargs
    )
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")
    model = sf.model.ParameterizedModel(
        structure,
        coords=dict(
            sample=metagenotypes.sample.values,
            position=metagenotypes.position.values,
            allele=metagenotypes.allele.values,
            strain=range(nstrain),
        ),
        hyperparameters=hyperparameters,
        data=condition_on,
        device=device,
        dtype=dtype,
    ).condition(**metagenotypes.to_counts_and_totals())

    start_time = time.time()
    est0, history0 = _estimate_parameters(model)
    _info("Finished initial fitting.")
    _info(f"Refitting missingness.")
    est1, history1 = _estimate_parameters(
        model.condition(
            pi=est0.data.communities.values,
            mu=est0.data.mu.values,
            alpha=est0.data.alpha.values,
            epsilon=est0.data.epsilon.values,
            m_hyper_r=est0.data.m_hyper_r.values,
        )
    )
    _info(f"Refitting genotypes.")
    est2, history2 = _estimate_parameters(
        model.condition(
            delta=est1.data.missingness.values,
            pi=est1.data.communities.values,
            mu=est1.data.mu.values,
            alpha=est1.data.alpha.values,
            epsilon=est1.data.epsilon.values,
            m_hyper_r=est1.data.m_hyper_r.values,
        ).with_hyperparameters(**stage2_hyperparameters)
    )
    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return (est0, est1), (history0, history1)


def fit_metagenotype_subsample_collapse_then_iteratively_refit_full_genotypes(
    structure,
    metagenotypes,
    nstrain,
    nposition,
    thresh,
    hyperparameters=None,
    stage2_hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
):
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}

    _estimate_parameters = lambda model: sf.estimation.estimate_parameters(
        model,
        quiet=quiet,
        **estimation_kwargs,
    )
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")
    _info(f"Fitting strain compositions using {nposition} randomly sampled positions.")
    metagenotypes_ss = metagenotypes.random_sample(nposition, "position")
    model = sf.model.ParameterizedModel(
        structure,
        coords=dict(
            sample=metagenotypes.sample.values,
            position=metagenotypes_ss.position.values,
            allele=metagenotypes.allele.values,
            strain=range(nstrain),
        ),
        hyperparameters=hyperparameters,
        data=condition_on,
        device=device,
        dtype=dtype,
    )

    start_time = time.time()
    est_curr, _ = _estimate_parameters(
        model.condition(**metagenotypes_ss.to_counts_and_totals())
    )
    _info(f"Finished initial fitting.")
    _info(f"Refitting genotypes with {stage2_hyperparameters}.")
    est_curr, _ = _estimate_parameters(
        model.with_hyperparameters(**stage2_hyperparameters)
        .condition(
            delta=est_curr.data.missingness.values,
            pi=est_curr.data.communities.values,
            mu=est_curr.data.mu.values,
            alpha=est_curr.data.alpha.values,
            epsilon=est_curr.data.epsilon.values,
            m_hyper_r=est_curr.data.m_hyper_r.values,
        )
        .condition(**metagenotypes_ss.to_counts_and_totals()),
    )
    _info(f"Collapsing {nstrain} initial strains.")
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est_curr,
        thresh=thresh,
        pdist_func=lambda w: w.genotypes.pdist(quiet=quiet),
    )
    _info(f"{agg_communities.sizes['strain']} strains after collapsing.")

    _info(f"Iteratively refitting missingness/genotypes.")
    chunks = {}
    for position_start, position_end in _chunk_start_end_iterator(
        metagenotypes.sizes["position"],
        nposition,
    ):
        _info(f"Fitting bin [{position_start}, {position_end}).")
        metagenotypes_chunk = metagenotypes.mlift(
            "isel", position=slice(position_start, position_end)
        )
        est_curr, _ = _estimate_parameters(
            model.with_amended_coords(
                position=metagenotypes_chunk.position.values,
                strain=agg_communities.strain.values,
            )
            .condition(
                pi=agg_communities.values,
                mu=est_curr.data.mu.values,
                alpha=est_curr.data.alpha.values,
                epsilon=est_curr.data.epsilon.values,
                m_hyper_r=est_curr.data.m_hyper_r.values,
            )
            .condition(**metagenotypes_chunk.to_counts_and_totals()),
        )
        est_curr, _ = _estimate_parameters(
            model.with_amended_coords(
                position=metagenotypes_chunk.position.values,
                strain=agg_communities.strain.values,
            )
            .with_hyperparameters(**stage2_hyperparameters)
            .condition(
                delta=est_curr.data.missingness.values,
                pi=est_curr.data.communities.values,
                mu=est_curr.data.mu.values,
                alpha=est_curr.data.alpha.values,
                epsilon=est_curr.data.epsilon.values,
                m_hyper_r=est_curr.data.m_hyper_r.values,
            )
            .condition(**metagenotypes_chunk.to_counts_and_totals()),
        )
        chunks[position_start] = est_curr
    est_curr = sf.data.World.concat(chunks, dim="position", rename_coords=False)
    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr


# def fit_then_relax_genotypes_and_collapse(
#     model,
#     world,
#     thresh,
#     initialize_params=None,
#     stage2_hyperparameters=None,
#     quiet=False,
#     **kwargs,
# ):
#     if stage2_hyperparameters is None:
#         stage2_hyperparameters = {}

#     est0, history0 = fit_simple(
#         model,
#         world,
#         initialize_params=initialize_params,
#         quiet=quiet,
#         **kwargs,
#     )
#     est1, history1 = fit_simple(
#         model.with_hyperparameters(**stage2_hyperparameters).condition(
#             pi=est0.data.communities.values,
#             mu=est0.data.mu.values,
#             alpha=est0.data.alpha.values,
#             epsilon=est0.data.epsilon.values,
#             m_hyper_r=est0.data.m_hyper_r.values,
#         ),
#         world,
#         quiet=quiet,
#         **kwargs,
#     )
#     agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
#         est1, thresh=thresh
#     )
#     est2, history2 = fit_simple(
#         model.with_amended_coords(strain=agg_communities.strain).condition(
#             pi=agg_communities.values,
#             mu=est1.data.mu.values,
#             alpha=est1.data.alpha.values,
#             epsilon=est1.data.epsilon.values,
#             m_hyper_r=est1.data.m_hyper_r.values,
#         ),
#         world,
#         quiet=quiet,
#         **kwargs,
#     )
#     est3, history3 = fit_simple(
#         model.with_amended_coords(strain=agg_communities.strain)
#         .with_hyperparameters(**stage2_hyperparameters)
#         .condition(
#             delta=est2.missingness.values,
#             pi=est2.communities.values,
#             mu=est2.data.mu.values,
#             alpha=est2.data.alpha.values,
#             epsilon=est2.data.epsilon.values,
#             m_hyper_r=est2.data.m_hyper_r.values,
#         ),
#         world,
#         quiet,
#         **kwargs,
#     )
#     return (est0, est1, est2, est3), (history0, history1, history2, history3)


# def fit_subsample_then_refit_relaxed_genotypes(
#     model,
#     world,
#     npositions,
#     stage2_hyperparameters,
#     initialize_params=None,
#     quiet=False,
#     **kwargs,
# ):
#     est0, history0 = fit_simple(model, world, initialize_params=initialize_params, quiet=quiet, **kwargs)
#     est1, history1 = fit_simple(
#         model.with_hyperparameters(**stage2_hyperparameters).condition(
#             # FIXME: Drop conditining on delta for consistency with 3-stage workflow?
#             delta=est0.data.missingness.values,
#             pi=est0.data.communities.values,
#             mu=est0.data.mu.values,
#             alpha=est0.data.alpha.values,
#             epsilon=est0.data.epsilon.values,
#             m_hyper_r=est0.data.m_hyper_r.values,
#         ),
#         world,
#         quiet=quiet,
#         **kwargs,
#     )
#     return (est0, est1), (history0, history1)


# def simulation_benchmark(
#     nsample,
#     nposition,
#     sim_nstrain,
#     fit_nstrain,
#     sim_model,
#     fit_model=None,
#     sim_data=None,
#     sim_hyperparameters=None,
#     sim_seed=None,
#     fit_data=None,
#     fit_hyperparameters=None,
#     fit_seed=None,
#     opt=pyro.optim.Adamax({"lr": 1e-0}, {"clip_norm": 100}),
#     **fit_kwargs,
# ):
#     if fit_model is None:
#         fit_model = sim_model

#     sim = sf.model.ParameterizedModel(
#         sim_model,
#         coords=dict(
#             sample=nsample,
#             position=nposition,
#             allele=["alt", "ref"],
#             strain=sim_nstrain,
#         ),
#         data=sim_data,
#         hyperparameters=sim_hyperparameters,
#     ).simulate_world(seed=sim_seed)

#     start_time = time.time()
#     est, history, *_ = sf.estimation.estimate_parameters(
#         sf.model.ParameterizedModel(
#             fit_model,
#             coords=dict(
#                 sample=nsample,
#                 position=nposition,
#                 allele=["alt", "ref"],
#                 strain=fit_nstrain,
#             ),
#             data=fit_data,
#             hyperparameters=fit_hyperparameters,
#         ).condition(
#             **sim.metagenotypes.to_counts_and_totals(),
#         ),
#         opt=opt,
#         seed=fit_seed,
#         **fit_kwargs,
#     )
#     end_time = time.time()

#     return (
#         sf.evaluation.weighted_genotype_error(sim, est),
#         sf.evaluation.community_error(sim, est),
#         end_time - start_time,
#     )
