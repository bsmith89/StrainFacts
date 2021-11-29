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
            "epsilon",
            "alpha",
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
        alpha_hyper_mean=200,
        alpha_hyper_scale=1,
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
    alpha_hyper_mean,
    alpha_hyper_scale,
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

    # Meta-community composition
    _rho = pyro.sample("_rho", dist.Dirichlet(_unit.repeat(s)))
    rho = pyro.deterministic(
        "rho", powerperturb_transformation(_rho, 1 / rho_hyper, _unit)
    )
    # rho = pyro.deterministic("rho", (_rho_unconditioned + eps) / (1 + eps * s))
    pyro.deterministic("metacommunity", rho)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        _pi = pyro.sample("_pi", dist.Dirichlet(_unit.repeat(s)))
        pi = pyro.deterministic(
            "pi", powerperturb_transformation(_pi, 1 / pi_hyper, rho),
        )
        epsilon = pyro.sample(
            "epsilon",
            dist.Beta(epsilon_hyper_spread, epsilon_hyper_spread / epsilon_hyper_mode),
        ).unsqueeze(-1)
        alpha = pyro.sample(
            "alpha",
            dist.LogNormal(loc=torch.log(alpha_hyper_mean), scale=alpha_hyper_scale),
        ).unsqueeze(-1)
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
    y = pyro.sample(
        "y",
        dist.BetaBinomial(
            concentration1=alpha * p, concentration0=alpha * (1 - p), total_count=m,
        ).to_event(),
    )
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
    pyro.deterministic("mu", m.mean(axis=1))
