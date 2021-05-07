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
    m_hyper_r=('sample', ),
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


def NegativeBinomialReparam(mu, r):
    p = 1.0 / ((r / mu) + 1.0)
    logits = torch.logit(p)
#     p = torch.clamp(p, eps, 1 - eps)
    return dist.NegativeBinomial(
        total_count=r,
        logits=logits,
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

def _pp_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp):
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
    return gamma, delta


def _gsm_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                'gamma',
                dist.RelaxedBernoulli(
                    temperature=gamma_hyper, probs=0.5,
                ),
            )
            delta = pyro.sample(
                'delta',
                dist.RelaxedBernoulli(
                    temperature=delta_hyper_temp, probs=delta_hyper_r
                ),
            )
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)
    return gamma, delta


def _beta_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                'gamma',
                dist.Beta(
                    gamma_hyper, gamma_hyper,
                ),
            )
            delta = pyro.sample(
                'delta',
                dist.RelaxedBernoulli(
                    delta_hyper_r * delta_hyper_temp,
                    (1 - delta_hyper_r) * delta_hyper_temp,
                ),
            )
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)
    return gamma, delta


def _hybrid_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample(
                "_gamma", dist.Beta(1., 1.)
            )
            gamma = pyro.deterministic(
                'gamma',
                unit_interval_power_transformation(_gamma, 1 / gamma_hyper, 1 / gamma_hyper))
#                 Position presence/absence
            delta = pyro.sample(
                'delta',
                dist.RelaxedBernoulli(
                    temperature=delta_hyper_temp, probs=delta_hyper_r
                ),
            )
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)
    return gamma, delta


def _dp_rho_module(s, rho_hyper):
#         # TODO: Will torch.ones(s) fail when I try to run this on the GPU because it's, by default on the CPU?
#         rho = pyro.sample(
#             "rho", dist.Dirichlet(rho_hyper * torch.ones(s))
#         )
    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    pyro.deterministic("metacommunity", rho)
    return rho


def _dirichlet_pi_epsilon_alpha_mu_module(
    n, pi_hyper, rho, epsilon_hyper_alpha, epsilon_hyper_beta, alpha_hyper_mean, alpha_hyper_scale, mu_hyper_mean, mu_hyper_scale
):
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
    return pi, epsilon, alpha, mu

def _dirichlet_pi_epsilon_alpha_r_mu_module(
    n, pi_hyper, rho, epsilon_hyper_alpha, epsilon_hyper_beta, alpha_hyper_mean, alpha_hyper_scale, mu_hyper_mean, mu_hyper_scale
):
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
    return pi, epsilon, alpha, mu


def _lognormal_alpha_hyper_mean_module(alpha_hyper_hyper_mean, alpha_hyper_hyper_scale):
    alpha_hyper_mean = pyro.sample(
        "alpha_hyper_mean",
        dist.LogNormal(
            loc=torch.log(alpha_hyper_hyper_mean), scale=alpha_hyper_hyper_scale
        ),
    )
    return alpha_hyper_mean


def _m_hyper_r_module(n, m_hyper_r_scale):
    m_hyper_r_mean = pyro.sample("m_hyper_r_mean", dist.LogNormal(loc=0.0, scale=10.0))
    m_hyper_r = pyro.sample("m_hyper_r", dist.LogNormal(loc=torch.log(m_hyper_r_mean), scale=m_hyper_r_scale).expand([n, 1]).to_event())
    return m_hyper_r


def _betabinomial_observation_module(
    pi, gamma, delta, m_hyper_r, mu, epsilon, alpha
):
    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    # TODO: Consider using pyro.distributions.GammaPoisson parameterization?
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            nu * mu.reshape((-1, 1)), m_hyper_r
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
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))


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
        mu_hyper_scale=10.0,
#         m_hyper_r_mu=1.,
        m_hyper_r_scale=1.,
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
#         m_hyper_r_mu,
        m_hyper_r_scale,
    ):
    gamma, delta = _pp_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp)
    rho = _dp_rho_module(s, rho_hyper)
    alpha_hyper_mean = _lognormal_alpha_hyper_mean_module(
        alpha_hyper_hyper_mean, alpha_hyper_hyper_scale
    )
    pi, epsilon, alpha, mu = _dirichlet_pi_epsilon_alpha_mu_module(
        n, pi_hyper, rho, epsilon_hyper_alpha, epsilon_hyper_beta, alpha_hyper_mean, alpha_hyper_scale, mu_hyper_mean, mu_hyper_scale
    )
    m_hyper_r = _m_hyper_r_module(n, m_hyper_r_scale)
    _betabinomial_observation_module(
        pi, gamma, delta, m_hyper_r, mu, epsilon, alpha
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
        mu_hyper_scale=10.0,
#         m_hyper_r_mu=1.,
        m_hyper_r_scale=1.,
        alpha_hyper_hyper_mean=100.0,
        alpha_hyper_hyper_scale=1.0,
        alpha_hyper_scale=0.5,
    ),
)
def gsm_fuzzy_missing_dp_betabinomial_metagenotype(
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
#         m_hyper_r_mu,
        m_hyper_r_scale,
    ):
    gamma, delta = _gsm_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp)
    rho = _dp_rho_module(s, rho_hyper)
    alpha_hyper_mean = _lognormal_alpha_hyper_mean_module(
        alpha_hyper_hyper_mean, alpha_hyper_hyper_scale
    )
    pi, epsilon, alpha, mu = _dirichlet_pi_epsilon_alpha_mu_module(
        n, pi_hyper, rho, epsilon_hyper_alpha, epsilon_hyper_beta, alpha_hyper_mean, alpha_hyper_scale, mu_hyper_mean, mu_hyper_scale
    )
    m_hyper_r = _m_hyper_r_module(n, m_hyper_r_scale)
    _betabinomial_observation_module(
        pi, gamma, delta, m_hyper_r, mu, epsilon, alpha
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
        mu_hyper_scale=10.0,
#         m_hyper_r_mu=1.,
        m_hyper_r_scale=1.,
        alpha_hyper_hyper_mean=100.0,
        alpha_hyper_hyper_scale=1.0,
        alpha_hyper_scale=0.5,
    ),
)
def hybrid_fuzzy_missing_dp_betabinomial_metagenotype(
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
#         m_hyper_r_mu,
        m_hyper_r_scale,
    ):
    gamma, delta = _hybrid_gamma_delta_module(s, g, gamma_hyper, delta_hyper_r, delta_hyper_temp)
    rho = _dp_rho_module(s, rho_hyper)
    alpha_hyper_mean = _lognormal_alpha_hyper_mean_module(
        alpha_hyper_hyper_mean, alpha_hyper_hyper_scale
    )
    pi, epsilon, alpha, mu = _dirichlet_pi_epsilon_alpha_mu_module(
        n, pi_hyper, rho, epsilon_hyper_alpha, epsilon_hyper_beta, alpha_hyper_mean, alpha_hyper_scale, mu_hyper_mean, mu_hyper_scale
    )
    m_hyper_r = _m_hyper_r_module(n, m_hyper_r_scale)
    _betabinomial_observation_module(
        pi, gamma, delta, m_hyper_r, mu, epsilon, alpha
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
        mu_hyper_scale=10.0,
        m_hyper_r_scale=1.,
        alpha_hyper_hyper_mean=100.0,
        alpha_hyper_hyper_scale=1.0,
        alpha_hyper_scale=0.5,
    ),
)
def scratch_metagenotype(
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
        epsilon_hyper_alpha,
        epsilon_hyper_beta,
        mu_hyper_mean,
        mu_hyper_scale,
        m_hyper_r_scale,
    ):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample(
                "_gamma", dist.Beta(1., 1.)
            )
            gamma = pyro.deterministic(
                'gamma',
                unit_interval_power_transformation(_gamma, 1 / gamma_hyper, 1 / gamma_hyper))
#                 Position presence/absence
            delta = pyro.sample(
                'delta',
                dist.RelaxedBernoulli(
                    temperature=delta_hyper_temp, probs=delta_hyper_r
                ),
            )
    pyro.deterministic("genotypes", gamma)
    pyro.deterministic("missingness", delta)
        
    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    pyro.deterministic("metacommunity", rho)
    
    alpha_hyper_mean = pyro.sample(
        "alpha_hyper_mean",
        dist.LogNormal(
            loc=torch.log(alpha_hyper_hyper_mean), scale=alpha_hyper_hyper_scale
        ),
    )
    m_hyper_r_mean = pyro.sample("m_hyper_r_mean", dist.LogNormal(loc=0.0, scale=10.0))
    
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
        m_hyper_r = pyro.sample(
            "m_hyper_r", dist.LogNormal(loc=torch.log(m_hyper_r_mean), scale=m_hyper_r_scale)
        ).unsqueeze(-1)
        # Sample coverage
        mu = pyro.sample(
            "mu", dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale)
        )
    pyro.deterministic("communities", pi)
    
    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    # TODO: Consider using pyro.distributions.GammaPoisson parameterization?
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            nu * mu.reshape((-1, 1)), m_hyper_r
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
        ).to_event(),
    )
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))


@sf.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        ['m', 'y',
         'genotypes', 'rho',
         'communities', 'metagenotypes']
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=0.01,
        pi_hyper=0.2,
    ),
)
def simple_metagenotype(
        n,
        g,
        s,
        a,
        gamma_hyper,
        rho_hyper,
        pi_hyper,
    ):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                "gamma", dist.Beta(gamma_hyper, gamma_hyper)
            )
    pyro.deterministic("genotypes", gamma)
    
    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    
    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(
            pi_hyper * rho,
            validate_args=False,
        ))
    pyro.deterministic("communities", pi)
    
    # Depth at each position
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            torch.tensor(100), torch.tensor(0.1),
        ).expand([n, g]).to_event(),
    )

    # Expected fractions of each allele at each position
    p = pyro.deterministic("p", pi @ gamma)

    # Observation
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
            validate_args=False,
        ).to_event(),
    )
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))


@sf.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        ['m', 'y',
         'genotypes', 'rho',
         'communities', 'metagenotypes']
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=0.01,
        pi_hyper=0.2,
    ),
)
def simple_metagenotype2(
        n,
        g,
        s,
        a,
        gamma_hyper,
        rho_hyper,
        pi_hyper,
    ):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            _gamma = pyro.sample(
                "_gamma", dist.Beta(1., 1.)
            )
            gamma = pyro.deterministic(
                'gamma',
                unit_interval_power_transformation(_gamma, 1 / gamma_hyper, 1 / gamma_hyper))
    pyro.deterministic("genotypes", gamma)
    
    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    
    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(
            pi_hyper * rho,
            validate_args=False,
        ))
    pyro.deterministic("communities", pi)
    
    # Depth at each position
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            torch.tensor(100), torch.tensor(0.1),
        ).expand([n, g]).to_event(),
    )

    # Expected fractions of each allele at each position
    p = pyro.deterministic("p", pi @ gamma)

    # Observation
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
        ).to_event(),
    )
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))


@sf.model.structure(
    dims=SHARED_DIMS,
    description=_mapping_subset(
        SHARED_DESCRIPTIONS,
        ['m', 'y', 'epsilon',
         'genotypes', 'rho',
         'communities', 'metagenotypes']
    ),
    default_hyperparameters=dict(
        gamma_hyper=0.01,
        rho_hyper=0.01,
        pi_hyper=0.2,
        epsilon_hyper_alpha=1.5,
        epsilon_hyper_beta=1.5 / 0.01,
    ),
)
def simple_metagenotype_plus_error(
        n,
        g,
        s,
        a,
        gamma_hyper,
        rho_hyper,
        pi_hyper,
        epsilon_hyper_alpha,
        epsilon_hyper_beta
    ):
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                "gamma", dist.Beta(gamma_hyper, gamma_hyper)
            )
    pyro.deterministic("genotypes", gamma)
    
    # Meta-community composition
    rho_betas = pyro.sample('rho_betas', dist.Beta(1., rho_hyper).expand([s - 1]).to_event())
    rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    
    with pyro.plate("sample", n, dim=-1):
        # Community composition
        pi = pyro.sample("pi", dist.Dirichlet(
            pi_hyper * rho,
            validate_args=False,
        ))
        epsilon = pyro.sample(
            "epsilon", dist.Beta(epsilon_hyper_alpha, epsilon_hyper_beta)
        ).unsqueeze(-1)
    pyro.deterministic("communities", pi)
    
    # Depth at each position
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
            torch.tensor(100), torch.tensor(0.1),
        ).expand([n, g]).to_event(),
    )

    # Expected fractions of each allele at each position
    p = pyro.deterministic("p", pi @ gamma)

    # Observation
    y = pyro.sample(
        "y",
        dist.Binomial(
            probs=p,
            total_count=m,
            validate_args=False,
        ).to_event(),
    )
    metagenotypes = pyro.deterministic("metagenotypes", torch.stack([y, m - y], dim=-1))

