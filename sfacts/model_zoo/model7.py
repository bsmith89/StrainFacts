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
    text_summary="""Metagenotype model intended for inference. Includes missingness.

No explicit error and no overdispersion of counts.

LogSoftTriangle prior on genotypes and introduces
a second tuning parameter for the sparsity of the (meta)community.

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
            "missingness",
            "community",
            "metagenotype",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=1e-10,
        delta_hyper_prob=0.9,
        delta_hyper_temp=1e-20,
        rho_hyper=0.2,
        rho_hyper2=1.0,
        pi_hyper=0.5,
        pi_hyper2=0.9,
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
    delta_hyper_prob,
    delta_hyper_temp,
    rho_hyper,
    rho_hyper2,
    pi_hyper,
    pi_hyper2,
    mu_hyper_mean,
    mu_hyper_scale,
    m_hyper_concentration,
    _unit,
):

    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                "gamma",
                LogSoftTriangle(
                    a=torch.log(gamma_hyper),
                    b=torch.log(gamma_hyper),
                ),
            )
            delta = pyro.sample(
                "delta",
                LogSoftTriangle(
                    a=torch.log(delta_hyper_prob * delta_hyper_temp),
                    b=torch.log((1 - delta_hyper_prob) * delta_hyper_temp),
                ),
            )
    pyro.deterministic("genotype", gamma)
    pyro.deterministic("missingness", delta)

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
    pyro.deterministic("community", pi)

    nu = pyro.deterministic("nu", pi @ delta)
    m = pyro.sample(
        "m",
        dist.GammaPoisson(
            rate=m_hyper_concentration / (nu * mu.reshape((-1, 1))),
            concentration=m_hyper_concentration,
        ).to_event(),
    )

    # Expected fractions of each allele at each position
    p = pyro.deterministic(
        "p",
        torch.clamp(
            pi @ (gamma * delta) / nu,
            min=0,
            max=1,
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
