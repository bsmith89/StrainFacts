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
from sfacts.logging_util import info



OPTIMIZERS = dict()
for _name, _default_optimizer_kwargs in [
    ("Adam", dict(optim_args={})),
    ("Adamax", dict(optim_args={"lr": 1e-2})),
    ("Adadelta", dict(optim_args={})),
    ("Adagrad", dict(optim_args={})),
    ("AdamW", dict(optim_args={})),
    ("RMSprop", dict(optim_args={"lr": 1e-2})),
]:
    OPTIMIZERS[_name] = pyro.optim.__dict__[_name], _default_optimizer_kwargs


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
    quiet=False,
    ignore_jit_warnings=False,
    seed=None,
    catch_keyboard_interrupt=False,
):
    if initialize_params is None:
        initialize_params = {}

    if jit:
        loss = pyro.infer.JitTrace_ELBO(ignore_jit_warnings=ignore_jit_warnings)
    else:
        loss = pyro.infer.Trace_ELBO()

    optimizer, default_optimizer_kwargs = OPTIMIZERS[optimizer_name]
    _optimizer_kwargs = default_optimizer_kwargs
    if not optimizer_kwargs is None:
        _optimizer_kwargs.update(optimizer_kwargs)
    opt = optimizer(**_optimizer_kwargs)
    if not quiet:
        info(f"Optimizing parameters with {optimizer_name}(**{_optimizer_kwargs})")

    sf.pyro_util.set_random_seed(seed, warn=(not quiet))

    _guide = pyro.infer.autoguide.AutoLaplaceApproximation(
        model,
        init_loc_fn=pyro.infer.autoguide.initialization.init_to_value(
            values=all_torch(**initialize_params, dtype=dtype, device=device)
        ),
    )
    svi = pyro.infer.SVI(model, _guide, opt, loss=loss)
    pyro.clear_param_store()

    history = []
    pbar = tqdm(range(maxiter), disable=quiet, mininterval=1.0)
    try:
        for i in pbar:
            elbo = svi.step()

            if np.isnan(elbo):
                pbar.close()
                raise RuntimeError("ELBO NaN?")

            # Fit tracking
            history.append(elbo)

            # Reporting/Breaking
            if i % 10 == 0:
                if i > lagB:
                    delta = history[-2] - history[-1]
                    delta_lagA = (history[-lagA] - history[-1]) / lagA
                    delta_lagB = (history[-lagB] - history[-1]) / lagB
                    pbar.set_postfix(
                        {
                            "ELBO": history[-1],
                            "delta": delta,
                            f"lag{lagA}": delta_lagA,
                            f"lag{lagB}": delta_lagB,
                        }
                    )
                    if (delta_lagA <= 0) and (delta_lagB <= 0):
                        pbar.close()
                        #                         info("Converged", quiet=quiet)
                        break
    except KeyboardInterrupt as err:
        pbar.close()
        info("Interrupted", quiet=quiet)
        if catch_keyboard_interrupt:
            pass
        else:
            raise err
    est = pyro.infer.Predictive(model, guide=_guide, num_samples=1)()
    est = {k: est[k].detach().cpu().numpy().mean(0).squeeze() for k in est.keys()}

    if device.startswith("cuda"):
        #         info(
        #             "CUDA available mem: {}".format(
        #                 torch.cuda.get_device_properties(0).total_memory
        #             ),
        #         )
        #         info("CUDA reserved mem: {}".format(torch.cuda.memory_reserved(0)))
        #         info("CUDA allocated mem: {}".format(torch.cuda.memory_allocated(0)))
        #         info(
        #             "CUDA free mem: {}".format(
        #                 torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        #             )
        #         )
        torch.cuda.empty_cache()

    return model.format_world(est), history


def strain_cluster(world, thresh, linkage="complete", pdist_func=None):
    if pdist_func is None:
        pdist_func = lambda w: w.genotypes.pdist()

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
def communities_aggregated_by_strain_cluster(
    world, diss_thresh, frac_thresh=0.0, **kwargs
):
    clust = strain_cluster(world, thresh=diss_thresh, **kwargs)
    comms = (
        world.communities.to_pandas()
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
    return sf.data.Communities(comms)


def nmf_approximation(
    world,
    s,
    regularization="both",
    alpha=1.0,
    l1_ratio=1.0,
    tol=1e-4,
    max_iter=int(1e4),
    random_state=None,
    init="random",
    **kwargs,
):
    d = world.metagenotypes.to_series().unstack("sample")
    columns = d.columns
    index = d.index

    gamma0, pi0, _ = non_negative_factorization(
        d.values,
        n_components=s,
        regularization={
            "community": "components",
            "genotype": "transformation",
            "both": "both",
        }[regularization],
        alpha=alpha,
        l1_ratio=l1_ratio,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        init=init,
        **kwargs,
    )
    pi1 = (
        pd.DataFrame(pi0, columns=columns)
        .rename_axis(index="strain")
        .stack()
        .to_xarray()
    )
    gamma1 = (
        pd.DataFrame(gamma0, index=index)
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
                communities=pi3.transpose("sample", "strain"),
                genotypes=gamma3.sel(allele="alt").transpose("strain", "position"),
                metagenotypes=world.metagenotypes.data,
            )
        )
    )
    return approx
