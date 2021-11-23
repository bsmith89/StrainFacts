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
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        [
            "rho",
            "p",
            "mu",
            "m",
            "y",
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
            _gamma = pyro.sample(
                "_gamma",
                ShiftedScaledDirichlet(
                    _unit.repeat(a), _unit.repeat(a) / a, 1 / gamma_hyper
                ),
            )
            gamma = _gamma[..., 0]
    pyro.deterministic("genotypes", gamma)

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
    pyro.deterministic("communities", pi)

    m = pyro.sample(
        "m",
        dist.GammaPoisson(
            rate=m_hyper_concentration / mu.reshape((-1, 1)), concentration=m_hyper_concentration
        )
        .expand([n, g])
        .to_event(),
    )

    # Expected fractions of each allele at each position
    p = pyro.deterministic("p", pi @ gamma)
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
            validate_args=False,
        ).to_event(),
    )
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
