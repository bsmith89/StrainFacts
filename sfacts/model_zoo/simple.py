import sfacts as sf
from sfacts.model_zoo.components import (
    _mapping_subset,
    unit_interval_power_transformation,
    k_simplex_power_transformation,
    SHARED_DESCRIPTIONS,
    SHARED_DIMS,
)
import torch
import pyro
import pyro.distributions as dist


@sf.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "p",
            "m",
            "y",
            "genotypes",
            "communities",
            "metagenotypes",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=5.0,
        pi_hyper=0.2,
        m_hyper_concentration=0.01,
        m_hyper_rate=2,
        # alpha=1e3,
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
    m_hyper_concentration,
    m_hyper_rate,
    # alpha,
    _unit,
):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample("_gamma", dist.Beta(_unit, _unit))
            gamma = pyro.deterministic(
                "gamma",
                unit_interval_power_transformation(_gamma, gamma_hyper, gamma_hyper),
            )
    pyro.deterministic("genotypes", gamma)

    # Meta-community composition
    _rho = pyro.sample("_rho", dist.Dirichlet(_unit.repeat(s)))
    rho = pyro.deterministic("rho", k_simplex_power_transformation(_rho, rho_hyper))
    pyro.deterministic("metacommunity", rho)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        _pi = pyro.sample("_pi", dist.Dirichlet(_unit.repeat(s)))
        pi = pyro.deterministic(
            "pi", k_simplex_power_transformation(_pi * rho, pi_hyper)
        )
    pyro.deterministic("communities", pi)

    m = pyro.sample(
        "m",
        dist.GammaPoisson(concentration=m_hyper_concentration, rate=m_hyper_rate)
        .expand([n, g])
        .to_event(),
    )

    p = pyro.deterministic("p", pi @ gamma)
    # Observation
    # y = pyro.sample(
    #     "y",
    #     dist.BetaBinomial(
    #         concentration1=alpha * p,
    #         concentration0=alpha * (1 - p),
    #         total_count=m,
    #     ).to_event(),
    # )
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
            validate_args=False,
        ).to_event(),
    )
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
