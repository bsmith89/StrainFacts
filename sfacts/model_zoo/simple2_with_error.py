import sfacts as sf
from sfacts.model_zoo.components import (
    _mapping_subset,
    powerperturb_transformation_unit_interval,
    powerperturb_transformation,
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
            "epsilon_hyper",
            "genotypes",
            "communities",
            "metagenotypes",
            "mu",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=5.0,
        pi_hyper=0.2,
        m_hyper_concentration=0.01,
        m_hyper_rate=2,
        epsilon_hyper_mode=0.01,
        epsilon_hyper_spread=1.5,
        eps=1e-20,
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
    epsilon_hyper_mode,
    epsilon_hyper_spread,
    # alpha,
    eps,
    _unit,
):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            # gamma = pyro.sample("gamma", dist.Beta(gamma_hyper, gamma_hyper))
            _gamma = pyro.sample("_gamma", dist.Beta(_unit, _unit))
            gamma = pyro.deterministic(
                "gamma",
                powerperturb_transformation_unit_interval(
                    _gamma, 1 / gamma_hyper, _unit
                ),
            )
    pyro.deterministic("genotypes", gamma)

    # Meta-community composition
    rho = pyro.sample("rho", dist.Dirichlet(_unit.repeat(s) * rho_hyper))
    pyro.deterministic("metacommunity", rho)

    epsilon_hyper = pyro.sample(
        "epsilon_hyper",
        dist.Beta(epsilon_hyper_spread, epsilon_hyper_spread / epsilon_hyper_mode),
    )
    epsilon = pyro.deterministic("epsilon", epsilon_hyper)
    with pyro.plate("sample", n, dim=-1):
        # Community composition
        _pi = pyro.sample("_pi", dist.Dirichlet(_unit.repeat(s) * pi_hyper))
        pi = pyro.deterministic(
            "pi",
            powerperturb_transformation(_pi, _unit, rho),
        )
    pyro.deterministic("communities", pi)

    m = pyro.sample(
        "m",
        dist.GammaPoisson(concentration=m_hyper_concentration, rate=m_hyper_rate)
        .expand([n, g])
        .to_event(),
    )

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic("p_noerr", pi @ gamma)
    p = pyro.deterministic(
        "p", (1 - epsilon / 2) * (p_noerr) + (epsilon / 2) * (1 - p_noerr)
    )
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
            # validate_args=False,
        ).to_event(),
    )
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
    pyro.deterministic("mu", m.mean(axis=1))
