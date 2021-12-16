import sfacts as sf
import time
import torch
import xarray as xr
import numpy as np


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
    anneal_hyperparameters=None,
    annealiter=0,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
):
    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    if estimation_kwargs is None:
        estimation_kwargs = {}

    _info(
        f"START: Fitting {nstrain} strains with data shape {metagenotypes.sizes}.",
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
        device=device,
        dtype=dtype,
        anneal_hyperparameters=anneal_hyperparameters,
        annealiter=annealiter,
        **estimation_kwargs,
    )
    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est, history


def fit_subsampled_metagenotypes_then_collapse_and_iteratively_refit_genotypes(
    structure,
    metagenotypes,
    nstrain,
    nposition,
    npositionB=None,
    hyperparameters=None,
    anneal_hyperparameters=None,
    annealiter=0,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    seed=None,
    nmf_init=False,
    nmf_init_kwargs=None,
    estimation_kwargs=None,
    stage2_hyperparameters=None,
    diss_thresh=0.0,
    frac_thresh=0.0,
):

    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    if estimation_kwargs is None:
        estimation_kwargs = {}
    if stage2_hyperparameters is None:
        stage2_hyperparameters = {}
    if nmf_init_kwargs is None:
        nmf_init_kwargs = {}

    est_list = []
    history_list = []

    _info(f"START: Fitting data with shape {metagenotypes.sizes}.")

    sf.pyro_util.set_random_seed(seed, warn=(not quiet))
    nposition = min(nposition, metagenotypes.sizes["position"])
    if npositionB is None:
        npositionB = nposition
    else:
        npositionB = min(npositionB, metagenotypes.sizes["position"])

    metagenotypes_full = metagenotypes
    _info(f"Subsampling {nposition} positions.")
    metagenotypes = metagenotypes.random_sample(position=nposition)

    _info(f"Fitting {nstrain} strains with data shape {metagenotypes.sizes}.")
    start_time = time.time()
    if nmf_init:
        _info("Initializing with NMF (this may take a while if the data dimensions are large).")
        nmf_kwargs = dict(
            alpha=0.0,
            solver="cd",
            tol=1e-3,
            eps=1e-5,
        )
        nmf_kwargs.update(nmf_init_kwargs)
        approx = sf.estimation.nmf_approximation(
            metagenotypes.to_world(),
            s=nstrain,
            random_state=seed,
            **nmf_kwargs,
        )
        initialize_params = dict(
            gamma=approx.genotypes.values,
            pi=approx.communities.values,
        )
    else:
        initialize_params = None

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

    est_curr, history = sf.estimation.estimate_parameters(
        pmodel,
        quiet=quiet,
        device=device,
        dtype=dtype,
        anneal_hyperparameters=anneal_hyperparameters,
        annealiter=annealiter,
        seed=seed,
        initialize_params=initialize_params,
        **estimation_kwargs,
    )
    history_list.append(history)
    est_list.append(est_curr)
    _info("Finished initial fitting.")
    _info(
        "Average metagenotype error: {}".format(
            sf.evaluation.metagenotype_error(est_curr, est_curr)[0]
        )
    )

    _info(f"Refitting genotypes with {stage2_hyperparameters}.")
    est_curr, history = sf.estimation.estimate_parameters(
        (
            pmodel.with_hyperparameters(**stage2_hyperparameters)
            .condition(pi=est_curr.data.communities.values)
            .condition(**metagenotypes.to_counts_and_totals())
        ),
        # initialize_params=dict(gamma=est_curr.data.genotypes.values),
        quiet=quiet,
        device=device,
        dtype=dtype,
        seed=seed,
        **estimation_kwargs,
    )
    history_list.append(history)
    est_list.append(est_curr)
    _info(
        "Average metagenotype error: {}".format(
            sf.evaluation.metagenotype_error(est_curr, est_curr)[0]
        )
    )

    _info(f"Collapsing {nstrain} initial strains.")
    agg_communities = sf.estimation.communities_aggregated_by_strain_cluster(
        est_curr,
        diss_thresh=diss_thresh,
        pdist_func=lambda w: w.genotypes.pdist(quiet=quiet),
        frac_thresh=frac_thresh,
    )
    _info(f"{agg_communities.sizes['strain']} strains after collapsing.")

    _info("Iteratively refitting full-length genotypes.")
    genotypes_chunks = []
    for position_start, position_end in _chunk_start_end_iterator(
        metagenotypes_full.sizes["position"],
        npositionB,
    ):
        _info(f"Fitting bin [{position_start}, {position_end}).")
        metagenotypes_chunk = metagenotypes_full.mlift(
            "isel", position=slice(position_start, position_end)
        )
        est_curr, history = sf.estimation.estimate_parameters(
            pmodel.with_hyperparameters(**stage2_hyperparameters)
            .with_amended_coords(
                position=metagenotypes_chunk.position.values,
                strain=agg_communities.strain.values,
            )
            .condition(
                pi=agg_communities.values,
            )
            .condition(**metagenotypes_chunk.to_counts_and_totals()),
            quiet=quiet,
            device=device,
            dtype=dtype,
            seed=seed,
            **estimation_kwargs,
        )
        genotypes_chunks.append(est_curr.genotypes.data)
        history_list.append(history)
        est_list.append(est_curr)

    genotypes = sf.data.Genotypes(xr.concat(genotypes_chunks, dim="position"))
    est_curr = sf.data.World(
        est_curr.data.drop_dims(["position", "allele"]).assign(
            genotypes=genotypes.data,
            metagenotypes=metagenotypes_full.data,
        )
    )
    est_list.append(est_curr)

    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr, est_list, history_list

def fit_metagenotypes_complex(
    structure,
    metagenotypes,
    nstrain,
    hyperparameters=None,
    anneal_hyperparameters=None,
    annealiter=0,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    nmf_init=False,
    nmf_init_kwargs=None,
    nmf_seed=None,
    estimation_kwargs=None,
):

    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    if estimation_kwargs is None:
        estimation_kwargs = {}
    if nmf_init_kwargs is None:
        nmf_init_kwargs = {}

    est_list = []
    history_list = []

    _info(f"START: Fitting {nstrain} strains with data shape {metagenotypes.sizes}.")

    start_time = time.time()
    if nmf_init:
        _info("Initializing with NMF (this may take a while if the data is large).")
        nmf_kwargs = dict(
            alpha=0.0,
            solver="cd",
            tol=1e-2,
            eps=1e-5,
        )
        nmf_kwargs.update(nmf_init_kwargs)
        approx = sf.estimation.nmf_approximation(
            metagenotypes.to_world(),
            s=nstrain,
            random_state=nmf_seed,
            **nmf_kwargs,
        )
        initialize_params = dict(
            gamma=approx.genotypes.values,
            pi=approx.communities.values,
        )
    else:
        initialize_params = None

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

    est_curr, history = sf.estimation.estimate_parameters(
        pmodel,
        quiet=quiet,
        device=device,
        dtype=dtype,
        anneal_hyperparameters=anneal_hyperparameters,
        annealiter=annealiter,
        initialize_params=initialize_params,
        **estimation_kwargs,
    )
    history_list.append(history)
    est_list.append(est_curr)
    _info("Finished initial fitting.")
    _info(
        "Average metagenotype error: {}".format(
            sf.evaluation.metagenotype_error(est_curr, est_curr)[0]
        )
    )

    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr, est_list, history_list

def fit_genotypes_conditioned_on_communities_then_collapse(
    structure,
    metagenotypes,
    communities,
    hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
    diss_thresh=0.0,
    frac_thresh=0.0,
):

    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)

    if estimation_kwargs is None:
        estimation_kwargs = {}

    est_list = []
    history_list = []

    nposition = metagenotypes.sizes["position"]
    nsample = metagenotypes.sizes["sample"]
    nstrain = communities.sizes["strain"]
    _info(
        f"START: Fitting {nstrain} strains to metagenotypes with {nposition} positions and {nsample} samples."
    )

    start_time = time.time()
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
    ).condition(pi=communities.values, **metagenotypes.to_counts_and_totals())

    est_curr, history = sf.estimation.estimate_parameters(
        pmodel,
        quiet=quiet,
        device=device,
        dtype=dtype,
        **estimation_kwargs,
    )
    history_list.append(history)
    est_list.append(est_curr)
    _info("Finished model fitting.")
    _info(
        "Average metagenotype error: {}".format(
            sf.evaluation.metagenotype_error(est_curr, est_curr)[0]
        )
    )

    _info(f"Collapsing {nstrain} initial strains.")
    est_curr = sf.estimation.communities_aggregated_by_strain_cluster(
        est_curr,
        diss_thresh=diss_thresh,
        pdist_func=lambda w: w.genotypes.pdist(quiet=quiet),
        frac_thresh=frac_thresh,
    ).to_world()
    _info(f"{est_curr.sizes['strain']} strains after collapsing.")

    est_list.append(est_curr)
    history_list.append(np.array([]))

    end_time = time.time()
    delta_time = end_time - start_time
    _info(f"END: Fit in {delta_time} seconds.")
    return est_curr, est_list, history_list


def iteratively_fit_genotypes_conditioned_on_communities(
    structure,
    metagenotypes,
    communities,
    nposition,
    hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    quiet=False,
    estimation_kwargs=None,
):

    _info = lambda *args, **kwargs: sf.logging_util.info(*args, quiet=quiet, **kwargs)
    _phase_info = lambda *args, **kwargs: sf.logging_util.phase_info(*args, quiet=quiet, **kwargs)

    if estimation_kwargs is None:
        estimation_kwargs = {}

    est_list = []
    history_list = []

    nstrain = len(communities.strain)
    nsample = len(communities.sample)
    nposition_full = len(metagenotypes.position)
    with _phase_info("Fitting genotypes for {nposition_full} positions."):
        _info(f"Conditioned on provided communities with {nstrain} strains and {nsample} samples.")
        nposition = min(nposition, nposition_full)

        metagenotypes_full = metagenotypes
        start_time = time.time()
        pmodel = sf.model.ParameterizedModel(
            structure,
            coords=dict(
                sample=communities.sample.values,
                position=range(nposition),
                allele=metagenotypes_full.allele.values,
                strain=communities.strain.values,
            ),
            hyperparameters=hyperparameters,
            data=dict(
                pi=communities.values,
            ),
            device=device,
            dtype=dtype,
        )

        _info("Iteratively fitting genotypes by chunks.")
        genotypes_chunks = []
        for position_start, position_end in _chunk_start_end_iterator(
            metagenotypes_full.sizes["position"],
            nposition,
        ):
            with _phase_info(f"Chunk [{position_start}, {position_end})."):
                metagenotypes_chunk = metagenotypes_full.mlift(
                    "isel", position=slice(position_start, position_end)
                )
                est_curr, history = sf.estimation.estimate_parameters(
                    pmodel.with_amended_coords(
                        position=metagenotypes_chunk.position.values,
                    ).condition(**metagenotypes_chunk.to_counts_and_totals()),
                    quiet=quiet,
                    device=device,
                    dtype=dtype,
                    **estimation_kwargs,
                )
                genotypes_chunks.append(est_curr.genotypes.data)
                history_list.append(history)
                est_list.append(est_curr)

        with _phase_info(f"Concatenating chunks."):
            genotypes = sf.data.Genotypes(xr.concat(genotypes_chunks, dim="position"))
            est_curr = sf.data.World(
                est_curr.data.drop_dims(["position", "allele"]).assign(
                    genotypes=genotypes.data,
                    metagenotypes=metagenotypes_full.data,
                )
            )
        est_list.append(est_curr)
    return est_curr, est_list, history_list
