import sfacts as sf
from sfacts.model_zoo.components import (
    _mapping_subset,
    SHARED_DESCRIPTIONS,
    SHARED_DIMS,
    ShiftedScaledDirichlet,
    LogSoftTriangle,
)
import torch
import pyro
import pyro.distributions as dist


@sf.model.structure(
    text_summary="""Metagenotype model intended for inference, designed for simplicity and speed.

Just like model4 but with uniform sequencing error.

    """,
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "p",
            "mu",
            "epsilon",
            "m",
            "y",
            "genotype",
            "community",
            "metagenotype",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=1e-10,
        rho_hyper=0.2,
        rho_hyper2=1.0,
        pi_hyper=0.5,
        pi_hyper2=0.9,
        mu_hyper_mean=1.0,
        mu_hyper_scale=1.0,
        m_hyper_concentration=1.0,
        epsilon_hyper_mode=0.01,
        epsilon_hyper_spread=1.5,
    ),
)
def model(
    n,
    g,
    s,
    a,
    gamma_hyper,
    rho_hyper,
    rho_hyper2,
    pi_hyper,
    pi_hyper2,
    mu_hyper_mean,
    mu_hyper_scale,
    m_hyper_concentration,
    epsilon_hyper_mode,
    epsilon_hyper_spread,
    _unit,
):

    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                "gamma",
                LogSoftTriangle(a=torch.log(gamma_hyper), b=torch.log(gamma_hyper)),
            )
    pyro.deterministic("genotype", gamma)

    # Meta-community composition
    rho = pyro.sample(
        "rho", ShiftedScaledDirichlet(rho_hyper2.repeat(s), _unit, 1 / rho_hyper)
    )

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample(
            "pi", ShiftedScaledDirichlet(pi_hyper2.repeat(s), rho, 1 / pi_hyper)
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
        "p",
        torch.clamp(
            (1 - epsilon / 2) * (p_noerr) + (epsilon / 2) * (1 - p_noerr), min=0, max=1
        ),
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
