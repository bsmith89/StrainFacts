import sfacts as sf
from sfacts.model_zoo.components import (
    _mapping_subset,
    powerperturb_transformation_unit_interval,
    powerperturb_transformation,
    SHARED_DESCRIPTIONS,
    SHARED_DIMS,
    ShiftedScaledDirichlet,
)
import torch
import pyro
import pyro.distributions as dist


@sf.model.structure(
    text_summary="""Metagenotype model with fixed and uniform error, and sequencing depth.

Intended as a lowest-common-denomenator metagenotype simulator.
Used for ben benchmarks in the paper (Smith et al. 2022, Frontiers in Bioinformatics)

    """,
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "p",
            "p_noerr",
            "mu",
            "epsilon",
            "m",
            "genotype",
            "community",
            "metagenotype",
        ],
    ),
    default_hyperparameters=dict(
        pi_hyper=0.2,
        mu_hyper_mean=1.0,
        epsilon_hyper_mode=0.01,
    ),
)
def model(
    n,
    g,
    s,
    a,
    pi_hyper,
    mu_hyper_mean,
    epsilon_hyper_mode,
    _unit,
):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample("gamma", dist.Bernoulli(_unit * 0.5))
        genotype = pyro.deterministic("genotype", gamma)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(_unit.repeat(s) / s * pi_hyper))
    pyro.deterministic("community", pi)
    epsilon = pyro.deterministic("epsilon", _unit.repeat((n, 1)) * epsilon_hyper_mode)
    m = pyro.deterministic("m", _unit.repeat(n, g) * mu_hyper_mean)

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic("p_noerr", pi @ gamma)
    p = pyro.deterministic(
        "p", (1 - epsilon / 2) * (p_noerr) + (epsilon / 2) * (1 - p_noerr)
    )

    # Observation
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
        ).to_event(),
    )
    pyro.deterministic("metagenotype", torch.stack([y, m - y], dim=-1))
    pyro.deterministic("mu", _unit.repeat(n) * mu_hyper_mean)
