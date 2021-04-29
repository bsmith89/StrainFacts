import sfacts as sf
import pyro
import pyro.distributions as dist
import torch
from torch.nn.functional import pad as torch_pad


SHARED_DIMS = ('sample', 'position', 'strain', 'allele')
SHARED_DESCRIPTIONS = dict(
    gamma=('strain', 'position'),
    delta=('strain', 'position'),
    rho=('strain',),
    pi=('sample', 'strain'),
    epsilon=('sample',),
    m_hyper_r=(),
    mu=('sample',),
    nu=('sample', 'position'),
    p_noerr=('sample', 'position'),
    p=('sample', 'position'),
    alpha_hyper_mean=(),
    alpha=('sample',),
    m=('sample', 'position'),
    y=('sample', 'position'),
    genotypes=('strain', 'position'),
    missingness=('strain', 'position'),
    communities=('sample', 'strain'),
    metagenotypes=('sample', 'position', 'allele'),
)

def _mapping_subset(mapping, keys):
    return {k: mapping[k] for k in keys}

def stickbreaking_betas_to_probs(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return torch_pad(beta, (0, 1), value=1) * torch_pad(beta1m_cumprod, (1, 0), value=1)


def NegativeBinomialReparam(mu, r, eps):
    p = torch.clamp(1.0 / ((r / mu) + 1.0), eps, 1 - eps)
    return dist.NegativeBinomial(
        total_count=r,
        probs=p,
    )

def unit_interval_power_transformation(p, alpha, beta):
    log_p = torch.log(p)
    log_q = torch.log1p(-p)
    log_p_raised = log_p * alpha
    log_q_raised = log_q * beta
    return torch.exp(
        log_p_raised -
        torch.logsumexp(torch.stack([log_p_raised, log_q_raised]), dim=0)
    )

@sf.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        ['rho', 'epsilon', 'm_hyper_r', 'mu',
         'nu', 'p_noerr', 'p', 'm', 'y',
         'alpha_hyper_mean', 'alpha',
         'genotypes', 'missingness',
         'communities', 'metagenotypes']
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        delta_hyper_temp=0.01,
        delta_hyper_r=0.9,
        rho_hyper=5.0,
        pi_hyper=0.2,
        epsilon_hyper_alpha=1.5,
        epsilon_hyper_beta=1.5 / 0.01,
        mu_hyper_mean=1.0,
        mu_hyper_scale=1.0,
        m_hyper_r_mu=1.,
        m_hyper_r_scale=1.,
        m_eps=1e-5,
        alpha_hyper_hyper_mean=100.0,
        alpha_hyper_hyper_scale=1.0,
        alpha_hyper_scale=0.5,
    ),
)
def pp_fuzzy_missing_dp_betabinomial_metagenotype(
        n,
        g,
        s,
        a,
        gamma_hyper,
        delta_hyper_r,
        delta_hyper_temp,
        rho_hyper,  #=1.0,
        pi_hyper,  #=1.0,
        alpha_hyper_hyper_mean,  #=100.0,
        alpha_hyper_hyper_scale,  #=1.0,
        alpha_hyper_scale,  #=0.5,
        epsilon_hyper_alpha,  #=1.5,
        epsilon_hyper_beta,  #=1.5 / 0.01,
        mu_hyper_mean,  #=1.0,
        mu_hyper_scale,  #=1.0,
        m_hyper_r_mu,
        m_hyper_r_scale,
        m_eps,  #=1e-5,
    ):
    # Genotypes
    #     delta_hyper_p = pyro.sample('delta_hyper_p', dist.Beta(1., 1.))
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample(
                "_gamma", dist.Beta(1., 1.)
            )
            gamma = pyro.deterministic(
                'gamma',
                unit_interval_power_transformation(_gamma, 1 / gamma_hyper, 1 / gamma_hyper))
#                 Position presence/absence
            _delta = pyro.sample(
                "_delta", dist.Beta(1., 1.)
            )
            delta = pyro.deterministic(
                'delta',
                unit_interval_power_transformation(
                    _delta,
                    2 * (1 - delta_hyper_r) / delta_hyper_temp,
                    2 * delta_hyper_r / delta_hyper_temp
                )
            )

#                 delta = pyro.sample(
#                     'delta',
#                     dist.RelaxedBernoulli(
#                         temperature=delta_hyper_temp, probs=delta_hyper_p
#                     ),
#                 )

    # These deterministics are accessed by PointMixin class properties.
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)

    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))

#         # TODO: Will torch.ones(s) fail when I try to run this on the GPU because it's, by default on the CPU?
#         rho = pyro.sample(
#             "rho", dist.Dirichlet(rho_hyper * torch.ones(s))
#         )

    alpha_hyper_mean = pyro.sample(
        "alpha_hyper_mean",
        dist.LogNormal(
            loc=torch.log(alpha_hyper_hyper_mean), scale=alpha_hyper_hyper_scale
        ),
    )
    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(pi_hyper * rho, validate_args=False))
        # Sequencing error
        epsilon = pyro.sample(
            "epsilon", dist.Beta(epsilon_hyper_alpha, epsilon_hyper_beta)
        ).unsqueeze(-1)
        alpha = pyro.sample(
            "alpha",
            dist.LogNormal(loc=torch.log(alpha_hyper_mean), scale=alpha_hyper_scale),
        ).unsqueeze(-1)
        # Sample coverage
        mu = pyro.sample(
            "mu", dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale)
        )

    pyro.deterministic("communities", pi)

    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    m_hyper_r = pyro.sample("m_hyper_r", dist.LogNormal(loc=m_hyper_r_mu, scale=m_hyper_r_scale))
    # TODO: Consider using pyro.distributions.GammaPoisson parameterization?
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            # TODO: Is m_eps = 1e-5 problematic?
            nu * mu.reshape((-1, 1)), m_hyper_r, eps=m_eps
        ).to_event(),
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
#             validate_args=False,
        ).to_event(),
    )

    # TODO: Check that dim=0 works?
    pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))
