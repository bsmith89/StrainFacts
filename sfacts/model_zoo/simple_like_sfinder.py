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
        ["p", "m", "y", "genotypes", "communities", "metagenotypes", "mu",],
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
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
            _gamma = pyro.sample("_gamma", dist.Beta(_unit, _unit))
            gamma = pyro.deterministic(
                "gamma",
                powerperturb_transformation_unit_interval(
                    _gamma, 1 / gamma_hyper, _unit
                ),
            )
    pyro.deterministic("genotypes", gamma)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(_unit.repeat(s)))
    pyro.deterministic("communities", pi)

    m = pyro.sample(
        "m",
        dist.GammaPoisson(concentration=m_hyper_concentration, rate=m_hyper_rate)
        .expand([n, g])
        .to_event(),
    )

    epsilon = pyro.sample(
        "epsilon",
        dist.Beta(epsilon_hyper_spread, epsilon_hyper_spread / epsilon_hyper_mode),
    )

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic("p_noerr", pi @ gamma)
    p = pyro.deterministic(
        "p", (1 - epsilon / 2) * (p_noerr) + (epsilon / 2) * (1 - p_noerr)
    )
    # Observation
    y = pyro.sample(
        "y", dist.Binomial(probs=p, total_count=m, validate_args=False,).to_event(),
    )
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
    pyro.deterministic("mu", m.mean(axis=1))
