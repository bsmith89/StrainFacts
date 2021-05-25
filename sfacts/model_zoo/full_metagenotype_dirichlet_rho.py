import sfacts.model
from sfacts.model_zoo.components import (
    _mapping_subset,
    unit_interval_power_transformation,
    stickbreaking_betas_to_probs,
    NegativeBinomialReparam,
    SHARED_DESCRIPTIONS,
    SHARED_DIMS,
)
import torch
import pyro
import pyro.distributions as dist


@sfacts.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "epsilon",
            "m_hyper_r_mean",
            "m_hyper_r_scale",
            "m_hyper_r",
            "mu",
            "nu",
            "p_noerr",
            "p",
            "m",
            "y",
            "alpha_hyper_mean",
            "alpha",
            "genotypes",
            "missingness",
            "communities",
            "metagenotypes",
        ],
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        delta_hyper_temp=0.01,
        delta_hyper_r=0.9,
        rho_hyper=5.0,
        pi_hyper=0.2,
        mu_hyper_mean=1.0,
        mu_hyper_scale=10.0,
        epsilon_hyper_mode=0.01,
        epsilon_hyper_spread=1.5,
        alpha_hyper_hyper_mean=100.0,
        alpha_hyper_hyper_scale=1.0,
        alpha_hyper_scale=0.5,
    ),
)
def full_metagenotype_dirichlet_rho_model_structure(
    n,
    g,
    s,
    a,
    gamma_hyper,
    delta_hyper_r,
    delta_hyper_temp,
    rho_hyper,
    pi_hyper,
    alpha_hyper_hyper_mean,
    alpha_hyper_hyper_scale,
    alpha_hyper_scale,
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
                unit_interval_power_transformation(
                    _gamma, 1 / gamma_hyper, 1 / gamma_hyper
                ),
            )
            # Position presence/absence
            delta = pyro.sample(
                "delta",
                dist.RelaxedBernoulli(
                    temperature=delta_hyper_temp, probs=delta_hyper_r
                ),
            )
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)

    # Meta-community composition
    rho = pyro.sample("rho", dist.Dirichlet(_unit.repeat(s) * rho_hyper))
    pyro.deterministic("metacommunity", rho)

    alpha_hyper_mean = pyro.sample(
        "alpha_hyper_mean",
        dist.LogNormal(
            loc=torch.log(alpha_hyper_hyper_mean),
            scale=alpha_hyper_hyper_scale,
        ),
    )
    m_hyper_r_mean = pyro.sample("m_hyper_r_mean", dist.LogNormal(loc=_unit * 0.0, scale=_unit * 10.))
    m_hyper_r_scale = pyro.sample(
        "m_hyper_r_scale", dist.LogNormal(loc=_unit * 0.0, scale=_unit * 10.)
    )

    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(pi_hyper * rho, validate_args=False))
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
    # TODO: Consider using pyro.distributions.GammaPoisson parameterization?
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
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
