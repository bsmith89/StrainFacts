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
    text_summary="""Metagenotype model harnessing the shifted-scaled Dirichlet distribution.

This is the StrainFacts model used in the paper (Smith et al. 2022, Frontiers in Bioinformatics).

ShiftedScaledDirichlet distributions are used for each of genotype, metacommunity, and community.
The model includes both error and overdispersion parameters of which may be different for each sample.

    """,
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "p",
            "mu",
            "epsilon",
            "alpha",
            "m",
            "y",
            "genotype",
            "community",
            "metagenotype",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=1e-10,
        rho_hyper=0.5,
        pi_hyper=0.3,
        mu_hyper_mean=1.0,
        mu_hyper_scale=1.0,
        m_hyper_concentration=1.0,
        epsilon_hyper_mode=0.01,
        epsilon_hyper_spread=1.5,
        alpha_hyper_mean=10,
        alpha_hyper_scale=1e-6,
    ),
)
def model(
    n,
    g,
    s,
    a,
    gamma_hyper,
    rho_hyper,
    pi_hyper,
    mu_hyper_mean,
    mu_hyper_scale,
    epsilon_hyper_mode,
    epsilon_hyper_spread,
    alpha_hyper_mean,
    alpha_hyper_scale,
    m_hyper_concentration,
    _unit,
):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample(
                "_gamma",
                ShiftedScaledDirichlet(
                    _unit.repeat(a), _unit.repeat(a) / a, 1 / gamma_hyper
                ),
            )
            gamma = _gamma[..., 0]
    pyro.deterministic("genotype", gamma)

    # Meta-community composition
    rho = pyro.sample(
        "rho",
        ShiftedScaledDirichlet(_unit.repeat(s), _unit.repeat(s) / s, 1 / rho_hyper),
    )
    pyro.deterministic("metacommunity", rho)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample(
            "pi", ShiftedScaledDirichlet(_unit.repeat(s), rho, 1 / pi_hyper)
        )
        mu = pyro.sample(
            "mu", dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale)
        )
        epsilon = pyro.sample(
            "epsilon",
            dist.Beta(
                concentration1=epsilon_hyper_spread,
                concentration0=epsilon_hyper_spread / epsilon_hyper_mode,
            ),
        ).unsqueeze(-1)
        alpha = pyro.sample(
            "alpha",
            dist.LogNormal(loc=torch.log(alpha_hyper_mean), scale=alpha_hyper_scale),
        ).unsqueeze(-1)
    pyro.deterministic("community", pi)

    m = pyro.sample(
        "m",
        dist.GammaPoisson(
            rate=m_hyper_concentration / mu.reshape((-1, 1)),
            concentration=m_hyper_concentration,
        )
        .expand([n, g])
        .to_event(),
    )

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic("p_noerr", pi @ gamma)
    p = pyro.deterministic(
        "p", (1 - epsilon / 2) * (p_noerr) + (epsilon / 2) * (1 - p_noerr)
    )

    # Observation
    y = pyro.sample(
        "y",
        dist.BetaBinomial(
            concentration1=alpha * p,
            concentration0=alpha * (1 - p),
            total_count=m,
        ).to_event(),
    )
    pyro.deterministic("metagenotype", torch.stack([y, m - y], dim=-1))
