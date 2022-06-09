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

No explicit error and no overdispersion of counts.

Used the LogSoftTriangle prior on genotypes.

    """,
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "p",
            "mu",
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
        pi_hyper=0.5,
        mu_hyper_mean=1.0,
        mu_hyper_scale=1.0,
        m_hyper_concentration=1.0,
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
    m_hyper_concentration,
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
    rho = pyro.sample("rho", dist.Dirichlet(rho_hyper.repeat(s)))

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample(
            "pi", ShiftedScaledDirichlet(_unit.repeat(s), rho, 1 / pi_hyper)
        )
        mu = pyro.sample(
            "mu", dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale)
        )
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
    p = pyro.deterministic("p", torch.clamp(pi @ gamma, min=0, max=1))

    # Observation
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
        ).to_event(),
    )
    pyro.deterministic("metagenotype", torch.stack([y, m - y], dim=-1))
