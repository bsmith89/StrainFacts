import sfacts as sf
import time
import torch
import xarray as xr
import numpy as np
import pandas as pd
import logging


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


def fit_metagenotype_complex(
    structure,
    metagenotype,
    nstrain,
    hyperparameters=None,
    anneal_hyperparameters=None,
    annealiter=0,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    init_from=None,
    init_vars=["genotype", "community"],
    nmf_seed=None,
    estimation_kwargs=None,
):

    if estimation_kwargs is None:
        estimation_kwargs = {}

    if init_from:
        logging.info(f"Initializing {init_vars} with provided values.")
        initialize_params = {}
        if "genotype" in init_vars:
            assert (init_from.genotype.position == metagenotype.position).all()
            initialize_params["gamma"] = init_from.genotype.values
        if "community" in init_vars:
            assert (init_from.community.sample == metagenotype.sample).all()
            initialize_params["pi"] = init_from.community.values
    else:
        initialize_params = None

    est_list = []
    history_list = []

    with sf.logging_util.phase_info(
        f"Fitting {nstrain} strains with data shape {metagenotype.sizes}."
    ):
        pmodel = sf.model.ParameterizedModel(
            structure,
            coords=dict(
                sample=metagenotype.sample.values,
                position=metagenotype.position.values,
                allele=metagenotype.allele.values,
                strain=range(nstrain),
            ),
            hyperparameters=hyperparameters,
            data=condition_on,
            device=device,
            dtype=dtype,
        ).condition(**metagenotype.to_counts_and_totals())

        est, history = sf.estimation.estimate_parameters(
            pmodel,
            device=device,
            dtype=dtype,
            anneal_hyperparameters=anneal_hyperparameters,
            annealiter=annealiter,
            initialize_params=initialize_params,
            **estimation_kwargs,
        )
    logging.info(
        "Average metagenotype error: {}".format(
            sf.evaluation.metagenotype_error2(est, metagenotype, discretized=True)[0]
        )
    )

    return est, history


def iteratively_fit_genotype_conditioned_on_community(
    structure,
    metagenotype,
    community,
    nposition,
    hyperparameters=None,
    condition_on=None,
    device="cpu",
    dtype=torch.float32,
    estimation_kwargs=None,
):

    if estimation_kwargs is None:
        estimation_kwargs = {}

    est_list = []
    history_list = []

    nstrain = len(community.strain)
    nsample = len(community.sample)
    nposition_full = len(metagenotype.position)
    with sf.logging_util.phase_info(
        f"Fitting genotype for {nposition_full} positions."
    ):
        logging.info(
            f"Conditioned on provided community with {nstrain} strains and {nsample} samples."
        )
        nposition = min(nposition, nposition_full)

        metagenotype_full = metagenotype
        start_time = time.time()
        pmodel = sf.model.ParameterizedModel(
            structure,
            coords=dict(
                sample=community.sample.values,
                position=range(nposition),
                allele=metagenotype_full.allele.values,
                strain=community.strain.values,
            ),
            hyperparameters=hyperparameters,
            data=dict(
                pi=community.values,
            ),
            device=device,
            dtype=dtype,
        )

        logging.info("Iteratively fitting genotype by chunks.")
        genotype_chunks = []
        for position_start, position_end in _chunk_start_end_iterator(
            metagenotype_full.sizes["position"],
            nposition,
        ):
            with sf.logging_util.phase_info(
                f"Chunk [{position_start}, {position_end})."
            ):
                metagenotype_chunk = metagenotype_full.mlift(
                    "isel", position=slice(position_start, position_end)
                )
                est_curr, history = sf.estimation.estimate_parameters(
                    pmodel.with_amended_coords(
                        position=metagenotype_chunk.position.values,
                    ).condition(**metagenotype_chunk.to_counts_and_totals()),
                    device=device,
                    dtype=dtype,
                    **estimation_kwargs,
                )
                genotype_chunks.append(est_curr.genotype.data)
                history_list.append(history)
                est_list.append(est_curr)

        with sf.logging_util.phase_info(f"Concatenating chunks."):
            genotype = sf.data.Genotype(xr.concat(genotype_chunks, dim="position"))
            est_curr = sf.data.World(
                est_curr.data.drop_dims(["position", "allele"]).assign(
                    genotype=genotype.data,
                    metagenotype=metagenotype_full.data,
                )
            )
        est_list.append(est_curr)
    return est_curr, est_list, history_list


def evaluate_fit_against_metagenotype(ref, fit):
    # Re-indexing the simulation by the subset of positions and samples
    # that were actually fit.
    ref = ref.sel(position=fit.position.astype(int), sample=fit.sample.astype(int))

    mgen_error = sf.evaluation.metagenotype_error2(fit, discretized=True)
    # mgen_unifrac_discordance = sf.evaluation.mgen_unifrac_discordance(ref, fit)
    return dict(
        mgen_error=mgen_error[0],
        # mgen_unifrac_discordance=mgen_unifrac_discordance[0],
    )


def evaluate_fit_against_simulation(sim, fit):
    # Re-indexing the simulation by the subset of positions and samples
    # that were actually fit.
    sim = sim.sel(position=fit.position.astype(int), sample=fit.sample.astype(int))

    fwd_genotype_error = sf.evaluation.discretized_weighted_genotype_error(sim, fit)
    rev_genotype_error = sf.evaluation.discretized_weighted_genotype_error(fit, sim)
    bc_error = sf.evaluation.braycurtis_error(sim, fit)
    unifrac_error = sf.evaluation.unifrac_error(sim, fit)
    entropy_error = sf.evaluation.community_entropy_error(sim, fit)

    return dict(
        fwd_genotype_error=fwd_genotype_error[0],
        rev_genotype_error=rev_genotype_error[0],
        bc_error=bc_error[0],
        unifrac_error=unifrac_error[0],
        entropy_error=entropy_error[0],
    )
