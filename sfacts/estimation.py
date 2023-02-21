from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import non_negative_factorization

import sfacts as sf
from sfacts.pandas_util import idxwhere

import xarray as xr

# from sklearn.decomposition import non_negative_factorization
# from sfacts.genotype import genotype_pdist, adjust_genotype_by_missing
from sfacts.pyro_util import all_torch
import pandas as pd
import numpy as np

# import scipy as sp
# from scipy.spatial.distance import squareform
import pyro

# import pyro.distributions as dist
import torch
from tqdm import tqdm
import logging

TQDM_NUM_FORMAT_STRING = "{:+#0.2e}"

_tqdm_format_num = TQDM_NUM_FORMAT_STRING.format


OPTIMIZERS = dict()
for _name, _default_optimizer_kwargs in [
    ("Adam", dict(lr=0.05)),
    ("Adamax", dict(lr=0.05)),
    ("Adadelta", dict(lr=0.05)),
    ("Adagrad", dict(lr=0.05)),
    ("AdamW", dict(lr=0.05)),
    ("RMSprop", dict(lr=0.05)),
    ("SGD", dict(lr=0.05)),
]:
    OPTIMIZERS[_name] = torch.optim.__dict__[_name], _default_optimizer_kwargs


def linear_annealing_schedule(
    start, end, annealing_steps, wait_steps=0, total_steps=None
):
    if total_steps is None:
        total_steps = annealing_steps
    final_steps = total_steps - (annealing_steps)
    assert wait_steps < annealing_steps
    return np.concatenate(
        [
            np.repeat(start, wait_steps),
            np.linspace(start, end, num=annealing_steps - wait_steps),
            np.repeat(end, final_steps),
        ]
    )


def log_annealing_schedule(start, end, annealing_steps, wait_steps=0, total_steps=None):
    if total_steps is None:
        total_steps = annealing_steps
    final_steps = total_steps - (annealing_steps)
    assert wait_steps < annealing_steps
    return np.concatenate(
        [
            np.repeat(start, wait_steps),
            np.logspace(
                np.log10(start), np.log10(end), num=annealing_steps - wait_steps
            ),
            np.repeat(end, final_steps),
        ]
    )


def invlinear_annealing_schedule(
    start, end, annealing_steps, wait_steps=0, total_steps=None
):
    if total_steps is None:
        total_steps = annealing_steps
    final_steps = total_steps - (annealing_steps)
    assert wait_steps < annealing_steps
    return np.concatenate(
        [
            np.repeat(start, wait_steps),
            1 / np.linspace(1 / start, 1 / end, num=annealing_steps - wait_steps),
            np.repeat(end, final_steps),
        ]
    )


def named_annealing_schedule(name, *args, **kwargs):
    return dict(
        linear=linear_annealing_schedule,
        log=log_annealing_schedule,
        invlinear=invlinear_annealing_schedule,
    )[name](*args, **kwargs)


def get_scheduled_optimization_stepper(
    model,
    guide,
    loss,
    optimizer_name,
    patience,
    cooldown,
    factor=0.5,
    optimizer_kwargs=None,
    optimizer_clip_kwargs=None,
):
    optimizer, default_optimizer_kwargs = OPTIMIZERS[optimizer_name]
    _optimizer_kwargs = default_optimizer_kwargs.copy()
    if optimizer_kwargs is not None:
        _optimizer_kwargs.update(optimizer_kwargs)

    # opt = pyro.optim.ReduceLROnPlateau(optimizer(**_optimizer_kwargs)
    scheduler = pyro.optim.ReduceLROnPlateau(
        dict(
            optimizer=optimizer,
            optim_args=_optimizer_kwargs,
            patience=patience,
            factor=factor,
            cooldown=cooldown,
            threshold=0,
            min_lr=0,
            eps=0,
        ),
        clip_args=optimizer_clip_kwargs,  # FIXME: This is just for testing purposes.
    )
    svi = pyro.infer.SVI(model, guide, scheduler, loss=loss)
    return svi, scheduler


def estimate_parameters(
    model,
    dtype=torch.float32,
    device="cpu",
    initialize_params=None,
    jit=True,
    maxiter=10000,
    lagA=20,
    lagB=100,
    optimizer_name="Adamax",
    optimizer_kwargs=None,
    optimizer_clip_kwargs=None,
    ignore_jit_warnings=False,
    seed=None,
    catch_keyboard_interrupt=False,
    anneal_hyperparameters=None,
    annealiter=0,
    lr_annealing_factor=0.5,
    minimum_lr=1e-6,
):
    if initialize_params is None:
        initialize_params = {}

    if anneal_hyperparameters is None:
        anneal_hyperparameters = {}
    anneal_hyperparameters = {
        k: torch.tensor(
            named_annealing_schedule(
                **anneal_hyperparameters[k],
                annealing_steps=annealiter,
                total_steps=maxiter,
            ),
            dtype=dtype,
            device=device,
        )
        for k in anneal_hyperparameters
    }
    final_anneal_hyperparameters = {
        k: anneal_hyperparameters[k][-1] for k in anneal_hyperparameters
    }
    model = model.with_passed_hyperparameters(*anneal_hyperparameters.keys())
    logging.debug("anneal_hyperparameters=%s", anneal_hyperparameters)

    sf.pyro_util.set_random_seed(seed)

    if jit:
        loss = pyro.infer.JitTrace_ELBO(ignore_jit_warnings=ignore_jit_warnings)
    else:
        loss = pyro.infer.Trace_ELBO()

    pyro.clear_param_store()
    guide = pyro.infer.autoguide.AutoLaplaceApproximation(
        model,
        init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(
            values=all_torch(**initialize_params, dtype=dtype, device=device)
        ),
    )

    svi, scheduler = get_scheduled_optimization_stepper(
        model,
        guide,
        loss,
        optimizer_name,
        factor=lr_annealing_factor,
        patience=lagB,
        cooldown=lagB,
        optimizer_kwargs=optimizer_kwargs,
        optimizer_clip_kwargs=optimizer_clip_kwargs,
    )

    history = []
    tqdm.format_num = staticmethod(
        _tqdm_format_num
    )  # Monkeypatch to format numbers better
    pbar = tqdm(
        zip(range(maxiter), *anneal_hyperparameters.values()),
        total=maxiter,
        mininterval=1.0,
        bar_format="{l_bar}{r_bar}",
        disable=(not logging.getLogger().isEnabledFor(logging.INFO)),
    )
    try:
        for i, *passed_hyperparameters in pbar:
            elbo = svi.step(*passed_hyperparameters)
            if i > annealiter:
                scheduler.step(elbo)

            if np.isnan(elbo):
                pbar.close()
                raise RuntimeError("NLP NaN?")

            # Fit tracking
            history.append(elbo)

            # Updating/Reporting/Breaking
            if i % lagA == 0:
                learning_rate = list(scheduler.optim_objs.values())[
                    0
                ].optimizer.param_groups[0]["lr"]
                delta = delta_lagA = delta_lagB = np.nan
                if i > 2:
                    delta = history[-2] - history[-1]
                if i > lagA:
                    delta_lagA = (history[-lagA] - history[-1]) / lagA
                if i > lagB:
                    delta_lagB = (history[-lagB] - history[-1]) / lagB
                pbar_postfix = {
                    "NLP": history[-1],
                    "delta": np.nan_to_num(delta),
                    # f"lag{lagA}": np.nan_to_num(delta_lagA),
                    f"lag{lagB}": np.nan_to_num(delta_lagB),
                    "lr": learning_rate,
                }
                pbar_postfix.update(
                    {
                        k: float(v.cpu().numpy())
                        for k, v in zip(
                            anneal_hyperparameters.keys(), passed_hyperparameters
                        )
                    }
                )
                pbar.set_postfix(pbar_postfix)
                # if (delta_lagA <= 0) and (delta_lagB <= 0):
                if learning_rate < minimum_lr:
                    pbar.close()
                    logging.info(f"Converged: NLP={elbo:.5e}")
                    break

        else:
            pbar.close()
            elbo = svi.evaluate_loss(*final_anneal_hyperparameters.values())
            logging.info(f"Reached maxiter: NLP={elbo:.5e}")
    except KeyboardInterrupt as err:
        pbar.close()
        elbo = svi.evaluate_loss(*final_anneal_hyperparameters.values())
        logging.info(f"Interrupted: NLP={elbo:.5e}")
        if catch_keyboard_interrupt:
            pass
        else:
            raise err

    est = pyro.infer.Predictive(model, guide=guide, num_samples=1)(
        *passed_hyperparameters
    )
    est = {k: est[k].detach().cpu().numpy().mean(0).squeeze() for k in est.keys()}

    if device.startswith("cuda"):
        #         logging.info(
        #             "CUDA available mem: {}".format(
        #                 torch.cuda.get_device_properties(0).total_memory
        #             ),
        #         )
        #         logging.info("CUDA reserved mem: {}".format(torch.cuda.memory_reserved(0)))
        #         logging.info("CUDA allocated mem: {}".format(torch.cuda.memory_allocated(0)))
        #         logging.info(
        #             "CUDA free mem: {}".format(
        #                 torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        #             )
        #         )
        torch.cuda.empty_cache()

    return (
        model.with_hyperparameters(
            **{
                k: passed_hyperparameters[i].detach().cpu().numpy()
                for i, k in enumerate(anneal_hyperparameters.keys())
            }
        ).format_world(est),
        history,
    )


def strain_cluster(world, thresh, linkage="complete", pdist_func=None):
    if pdist_func is None:
        pdist_func = lambda w: w.genotype.pdist()

    clust = pd.Series(
        AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=thresh,
            linkage="complete",
            affinity="precomputed",
        )
        .fit(pdist_func(world))
        .labels_,
        index=world.strain,
    )
    return clust


# TODO: Separate coverage-thresholding from clustering.
def community_aggregated_by_strain_cluster(
    world, diss_thresh, frac_thresh=0.0, **kwargs
):
    clust = strain_cluster(world, thresh=diss_thresh, **kwargs)
    comms = (
        world.community.to_pandas()
        .groupby(clust, axis="columns")
        .sum()
        .rename_axis(columns="strain")
    )
    low_max_frac_strains = idxwhere(comms.max() < frac_thresh)
    if len(low_max_frac_strains) > 0:
        comms[-1] = comms[low_max_frac_strains].sum(1)
    comms = comms.drop(columns=low_max_frac_strains)
    comms = comms.stack().to_xarray()
    comms = comms / comms.sum("strain")
    return sf.data.Community(comms)


def nmf_approximation(
    world,
    s,
    eps=0,
    **kwargs,
):
    d = (
        world.metagenotype
        # .frequencies(pseudo=pseudo)
        .to_series().unstack("sample")
    )
    columns = d.columns
    index = d.index

    gamma0, pi0, _ = non_negative_factorization(
        d.values,
        n_components=s,
        **kwargs,
    )
    pi1 = (
        pd.DataFrame(pi0 + eps, columns=columns)
        .rename_axis(index="strain")
        .stack()
        .to_xarray()
    )
    gamma1 = (
        pd.DataFrame(gamma0 + eps, index=index)
        .rename_axis(columns="strain")
        .stack()
        .to_xarray()
    )

    # Rebalance estimates: mean strain genotype of 1
    gamma1_strain_factor = gamma1.sum("allele").mean("position")
    gamma2 = gamma1 / gamma1_strain_factor
    pi2 = pi1 * gamma1_strain_factor

    # Transform estimates: sum-to-1
    gamma3 = (gamma2 / gamma2.sum("allele")).fillna(0.5)
    pi3 = pi2 / pi2.sum("strain")

    approx = sf.data.World(
        xr.Dataset(
            dict(
                community=pi3.transpose("sample", "strain"),
                genotype=gamma3.sel(allele="alt").transpose("strain", "position"),
                metagenotype=world.metagenotype.data,
            )
        )
    )
    return approx


def clust_approximation(
    world,
    s,
    thresh=None,
    pseudo=1.0,
    frac=1.0,
    **kwargs,
):
    mgen = world.metagenotype
    if thresh is None:
        clust = mgen.clusters(s_or_thresh=s, **kwargs)
    else:
        clust = mgen.clusters(s_or_thresh=thresh, **kwargs)

    nclust = len(clust.unique())
    logging.info(f"Clustering approximated {nclust} strains.")
    assert nclust < s, (
        "Clustering identified too many clusters (> allowed nstrains). "
        "Try either increasing the number of strains or the clustering threshold."
    )
    geno = sf.Metagenotype(
        mgen.to_series()
        .unstack("sample")
        .groupby(clust, axis="columns")
        .sum()
        .rename_axis(columns="sample")
        .reindex(columns=range(s), fill_value=0)
        .stack()
        .to_xarray()
        .transpose("sample", "position", "allele")
    ).to_estimated_genotype(pseudo=pseudo)
    geno.validate_constraints()
    comm = sf.Community(
        clust.to_frame(name="strain")
        .assign(community=frac)
        .reset_index()
        .set_index(["sample", "strain"])
        .squeeze()
        .unstack("strain")
        .reindex(columns=range(s))
        .fillna((1 - frac) / (s - 1))
        .stack()
        .to_xarray()
    )
    comm.validate_constraints()
    world = sf.World.from_combined(mgen, comm, geno)
    return world
