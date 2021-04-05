
from sfacts.pyro_util import as_torch, all_torch
import pyro
import pyro.distributions as dist
import torch
from functools import partial


def NegativeBinomialReparam(mu, r, eps=1e-5):
    p = torch.clamp(1. / ((r / mu) + 1.), min=eps, max=1. - eps)
    return dist.NegativeBinomial(
        total_count=r,
        probs=p
    )

def model(
    n,
    g,
    s,
    gamma_hyper=1.,
    delta_hyper_temp=0.1,
    delta_hyper_p=0.9,
    rho_hyper=1.,
    pi_hyper=1.,
    alpha_hyper_hyper_mean=100.,
    alpha_hyper_hyper_scale=1.,
    alpha_hyper_scale=0.5,
    epsilon_hyper_alpha=1.5,
    epsilon_hyper_beta=1.5 / 0.01,
    mu_hyper_mean=1.,
    mu_hyper_scale=1.,
    m_hyper_r=1.,
    m_eps=1e-5,
    dtype=torch.float32,
    device='cpu',
):
    (
        gamma_hyper,
        delta_hyper_temp,
        delta_hyper_p,
        rho_hyper,
        pi_hyper,
        alpha_hyper_hyper_mean,
        alpha_hyper_hyper_scale,
        alpha_hyper_scale,
        epsilon_hyper_alpha,
        epsilon_hyper_beta,
        mu_hyper_mean,
        mu_hyper_scale,
        m_hyper_r,
        m_eps,
    ) = (
        as_torch(x, dtype=dtype, device=device)
        for x in [
            gamma_hyper,
            delta_hyper_temp,
            delta_hyper_p,
            rho_hyper,
            pi_hyper,
            alpha_hyper_hyper_mean,
            alpha_hyper_hyper_scale,
            alpha_hyper_scale,
            epsilon_hyper_alpha,
            epsilon_hyper_beta,
            mu_hyper_mean,
            mu_hyper_scale,
            m_hyper_r,
            m_eps,
        ]
    )

    # Genotypes
    with pyro.plate('position', g, dim=-1):
        with pyro.plate('strain', s, dim=-2):
            gamma = pyro.sample(
                'gamma', dist.RelaxedBernoulli(temperature=gamma_hyper, logits=torch.tensor(0, dtype=dtype, device=device).squeeze())
            )
            # Position presence/absence
            delta = pyro.sample(
                'delta', dist.RelaxedBernoulli(temperature=delta_hyper_temp, probs=delta_hyper_p)
            )
    
    # Meta-community composition
    rho = pyro.sample('rho', dist.Dirichlet(rho_hyper * torch.ones(s, dtype=dtype, device=device)))

    alpha_hyper_mean = pyro.sample('alpha_hyper_mean', dist.LogNormal(loc=torch.log(alpha_hyper_hyper_mean), scale=alpha_hyper_hyper_scale))
    with pyro.plate('sample', n, dim=-1):
        # Community composition
        pi = pyro.sample('pi', dist.RelaxedOneHotCategorical(temperature=pi_hyper, probs=rho))
        # Sample coverage
        mu = pyro.sample('mu', dist.LogNormal(loc=torch.log(mu_hyper_mean), scale=mu_hyper_scale))
        # Sequencing error
        epsilon = pyro.sample('epsilon', dist.Beta(epsilon_hyper_alpha, epsilon_hyper_beta)).unsqueeze(-1)
        alpha = pyro.sample('alpha', dist.LogNormal(loc=torch.log(alpha_hyper_mean, ), scale=alpha_hyper_scale)).unsqueeze(-1)
        
    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    m = pyro.sample('m', NegativeBinomialReparam(nu * mu.reshape((-1,1)), m_hyper_r, m_eps).to_event())

    # Expected fractions of each allele at each position
    p_noerr = pyro.deterministic('p_noerr', pi @ (gamma * delta) / nu)
    p = pyro.deterministic('p',
        (1 - epsilon / 2) * (p_noerr) +
        (epsilon / 2) * (1 - p_noerr)
    )
    
    # Observation
    y = pyro.sample(
        'y',
        dist.BetaBinomial(
            concentration1=alpha * p,
            concentration0=alpha * (1 - p),
            total_count=m
        ).to_event(),
    )
    
def condition_model(model, data=None, device='cpu', dtype=torch.float32, **model_kwargs):
    if data is None:
        data = {}
        
    conditioned_model = partial(
        pyro.condition(
            model,
            data=all_torch(**data, dtype=dtype, device=device),
        ),
        **model_kwargs,
        dtype=dtype,
        device=device,
    )
    return conditioned_model
    
def simulate(model):
    obs = pyro.infer.Predictive(model, num_samples=1)()
    obs = {
        k: obs[k].detach().cpu().numpy().squeeze()
        for k in obs.keys()
    }
    return obs
