import sfacts as sf
import time
import torch
import xarray as xr


def _chunk_start_end_iterator(total, per):
    for i in range(total // per):
        yield (per * i), (per * (i + 1))
    if (i + 1) * per < total:
        yield (i + 1) * per, total


def simulate_world(
    structure,
    sizes,
    hyperparameters,
    seed=None,
    data=None,
    dtype=torch.float32,
    device="cpu",
):
    if data is None:
        data = {}

    assert len(sizes) == 3, "Sizes should only be for strain, sample, and position."
    model = sf.model.ParameterizedModel(
        structure=structure,
        coords=dict(
            strain=sizes["strain"],
            sample=sizes["sample"],
            position=sizes["position"],
            allele=["alt", "ref"],
        ),
        hyperparameters=hyperparameters,
        dtype=dtype,
        device=device,
        data=data,
    )
    world = model.simulate_world(seed=seed)

    return model, world


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
    pmodel = sf.model.ParameterizedModel(
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
        pmodel,
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

    _estimate_parameters = lambda pmodel: sf.estimation.estimate_parameters(
        pmodel, quiet=quiet, **estimation_kwargs
    )
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")
    pmodel = sf.model.ParameterizedModel(
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
    est0, history0 = _estimate_parameters(pmodel)
    _info("Finished initial fitting.")
    _info("Refitting missingness.")
    est1, history1 = _estimate_parameters(
        pmodel.condition(
            pi=est0.data.communities.values,
            mu=est0.data.mu.values,
            alpha=est0.data.alpha.values,
            epsilon=est0.data.epsilon.values,
            m_hyper_r=est0.data.m_hyper_r.values,
        )
    )
    _info("Refitting genotypes.")
    est2, history2 = _estimate_parameters(
        pmodel.condition(
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


def fit_subsampled_metagenotype_collapse_strains_then_iteratively_refit_full_genotypes(
    structure,
    metagenotypes,
    nstrain,
    nposition,
    diss_thresh,
    frac_thresh=0.0,
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

    _estimate_parameters = lambda pmodel: sf.estimation.estimate_parameters(
        pmodel,
        quiet=quiet,
        **estimation_kwargs,
    )
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    nposition = min(nposition, metagenotypes.sizes["position"])

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")
    _info(f"Fitting strain compositions using {nposition} randomly sampled positions.")
    metagenotypes_ss = metagenotypes.random_sample(nposition, "position")
    pmodel = sf.model.ParameterizedModel(
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
        pmodel.condition(**metagenotypes_ss.to_counts_and_totals())
    )
    _info("Finished initial fitting.")
    _info(f"Refitting genotypes with {stage2_hyperparameters}.")
    est_curr, _ = _estimate_parameters(
        pmodel.with_hyperparameters(**stage2_hyperparameters)
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
        diss_thresh=diss_thresh,
        pdist_func=lambda w: w.genotypes.pdist(quiet=quiet),
        frac_thresh=frac_thresh,
    )
    _info(f"{agg_communities.sizes['strain']} strains after collapsing.")

    _info("Iteratively refitting missingness/genotypes.")
    genotypes_chunks = []
    missingness_chunks = []
    for position_start, position_end in _chunk_start_end_iterator(
        metagenotypes.sizes["position"],
        nposition,
    ):
        _info(f"Fitting bin [{position_start}, {position_end}).")
        metagenotypes_chunk = metagenotypes.mlift(
            "isel", position=slice(position_start, position_end)
        )
        est_curr, _ = _estimate_parameters(
            pmodel.with_amended_coords(
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
            pmodel.with_amended_coords(
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
        genotypes_chunks.append(est_curr.genotypes.data)
        missingness_chunks.append(est_curr.missingness.data)

    genotypes = sf.data.Genotypes(xr.concat(genotypes_chunks, dim="position"))
    missingness = sf.data.Missingness(xr.concat(missingness_chunks, dim="position"))
    est_curr = sf.data.World(
        est_curr.data.drop_dims(["position", "allele"]).assign(
            genotypes=genotypes.data,
            missingness=missingness.data,
            metagenotypes=metagenotypes.data,
        )
    )
    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr


def fit_subsampled_metagenotype_collapse_strains_then_iteratively_refit_full_genotypes_no_missing(
    structure,
    metagenotypes,
    nstrain,
    nposition,
    diss_thresh,
    frac_thresh=0.0,
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

    _estimate_parameters = lambda pmodel: sf.estimation.estimate_parameters(
        pmodel,
        quiet=quiet,
        **estimation_kwargs,
    )
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    nposition = min(nposition, metagenotypes.sizes["position"])

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")
    _info(f"Fitting strain compositions using {nposition} randomly sampled positions.")
    metagenotypes_ss = metagenotypes.random_sample(nposition, "position")
    pmodel = sf.model.ParameterizedModel(
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
        pmodel.condition(**metagenotypes_ss.to_counts_and_totals())
    )
    _info("Finished initial fitting.")
    _info(f"Refitting genotypes with {stage2_hyperparameters}.")
    est_curr, _ = _estimate_parameters(
        pmodel.with_hyperparameters(**stage2_hyperparameters)
        .condition(
            pi=est_curr.data.communities.values,
            alpha=est_curr.data.alpha.values,
            epsilon=est_curr.data.epsilon.values,
        )
        .condition(**metagenotypes_ss.to_counts_and_totals()),
    )
    _info(f"Collapsing {nstrain} initial strains.")
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est_curr,
        diss_thresh=diss_thresh,
        pdist_func=lambda w: w.genotypes.pdist(quiet=quiet),
        frac_thresh=frac_thresh,
    )
    _info(f"{agg_communities.sizes['strain']} strains after collapsing.")

    _info(f"Iteratively refitting genotypes. {stage2_hyperparameters}")
    genotypes_chunks = []
    for position_start, position_end in _chunk_start_end_iterator(
        metagenotypes.sizes["position"],
        nposition,
    ):
        _info(f"Fitting bin [{position_start}, {position_end}).")
        metagenotypes_chunk = metagenotypes.mlift(
            "isel", position=slice(position_start, position_end)
        )
        est_curr, _ = _estimate_parameters(
            pmodel.with_amended_coords(
                position=metagenotypes_chunk.position.values,
                strain=agg_communities.strain.values,
            )
            .with_hyperparameters(**stage2_hyperparameters)
            .condition(
                pi=agg_communities.values,
                mu=est_curr.data.mu.values,
                alpha=est_curr.data.alpha.values,
                epsilon=est_curr.data.epsilon.values,
                m_hyper_r=est_curr.data.m_hyper_r.values,
            )
            .condition(**metagenotypes_chunk.to_counts_and_totals()),
        )
        genotypes_chunks.append(est_curr.genotypes.data)

    genotypes = sf.data.Genotypes(xr.concat(genotypes_chunks, dim="position"))
    est_curr = sf.data.World(
        est_curr.data.drop_dims(["position", "allele"]).assign(
            genotypes=genotypes.data,
            metagenotypes=metagenotypes.data,
        )
    )
    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr
