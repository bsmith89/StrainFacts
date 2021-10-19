import sfacts as sf
from sfacts.model_zoo.components import (
    _mapping_subset,
    powerperturb_transformation_unit_interval,
    powerperturb_transformation,
    NegativeBinomialReparam,
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
            "epsilon",
            "m_hyper_r",
            "mu",
            "nu",
            "p_noerr",
            "p",
            "m",
            "y",
            "alpha",
            "genotypes",
            "missingness",
            "communities",
            "metagenotypes",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=5.0,
        pi_hyper=0.2,
        mu_hyper_mean=1.0,
        mu_hyper_scale=10.0,
        epsilon_hyper_mode=0.01,
        epsilon_hyper_spread=1.5,
        m_hyper_r_mean=1.0,
        m_hyper_r_scale=1.0,
        alpha_hyper_mean=100.0,
        alpha_hyper_scale=0.5,
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
    alpha_hyper_mean,
    alpha_hyper_scale,
    m_hyper_r_mean,
    m_hyper_r_scale,
    mu_hyper_mean,
    mu_hyper_scale,
    epsilon_hyper_mode,
    epsilon_hyper_spread,
    _unit,
):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample("_gamma", dist.Beta(_unit, _unit))
            gamma = pyro.deterministic(
                "gamma",
                powerperturb_transformation_unit_interval(_gamma, 1 / gamma_hyper, _unit),
            )
            # # Position presence/absence
            # delta = pyro.sample(
            #     "delta",
            #     dist.RelaxedBernoulli(
            #         temperature=delta_hyper_temp, probs=delta_hyper_r
            #     ),
            # )
    delta = pyro.deterministic("delta", torch.ones_like(gamma) * _unit)
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)

    # Meta-community composition
    _rho = pyro.sample("_rho", dist.Dirichlet(_unit.repeat(s)))
    rho = pyro.deterministic("rho", powerperturb_transformation(_rho, 1 / rho_hyper, _unit))
    pyro.deterministic("metacommunity", rho)

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        _pi = pyro.sample("_pi", dist.Dirichlet(_unit.repeat(s)))
        pi = pyro.deterministic(
            "pi", powerperturb_transformation(_pi, 1 / pi_hyper, rho)
        )
        # Sequencing error
        epsilon = pyro.sample(
            "epsilon",
            dist.Beta(epsilon_hyper_spread, epsilon_hyper_spread / epsilon_hyper_mode),
        ).unsqueeze(-1)
        alpha = pyro.sample(
            "alpha",
            dist.LogNormal(loc=torch.log(alpha_hyper_mean), scale=alpha_hyper_scale),
        ).unsqueeze(-1)
        m_hyper_r = pyro.sample(
            "m_hyper_r",
            dist.LogNormal(loc=torch.log(m_hyper_r_mean), scale=m_hyper_r_scale),
        ).unsqueeze(-1)
        # Sample coverage
        mu = pyro.sample(
            "mu",
            dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale),
        )
    pyro.deterministic("communities", pi)

    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(nu * mu.reshape((-1, 1)), m_hyper_r).to_event(),
    )

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic("p_noerr", pi @ (gamma * delta) / nu)
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
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
