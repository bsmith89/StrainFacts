from sfacts.pyro_util import as_torch, all_torch
import pyro
import pyro.distributions as dist
import torch
from functools import partial
from torch.nn.functional import pad as torch_pad


def stickbreaking_betas_to_probs(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return torch_pad(beta, (0, 1), value=1) * torch_pad(beta1m_cumprod, (1, 0), value=1)


def stickbreaking_betas_to_probs2(beta):
    """I thought this might be more stable, but it turns out the gradient is NOT more stable."""
    log_beta1m = torch.log(1 - beta)
    log_beta1m_cumprod = log_beta1m.cumsum(-1)
    log_beta_pad = torch_pad(torch.log(beta), (0, 1), value=0)
    log_beta1m_cumprod_pad = torch_pad(log_beta1m_cumprod, (1, 0), value=0)
    #     beta1m_cumprod = (1 - beta).cumprod(-1)
    #     return torch_pad(beta, (0, 1), value=1) * torch_pad(beta1m_cumprod, (1, 0), value=1)
    return torch.exp(log_beta_pad + log_beta1m_cumprod_pad)


def NegativeBinomialReparam(mu, r, eps):
    p = torch.clamp(1.0 / ((r / mu) + 1.0), eps, 1 - eps)
    return dist.NegativeBinomial(
        total_count=r,
        probs=p,
    )


def model(
    n,
    g,
    s,
    gamma_hyper=1.0,
    delta_hyper_temp=0.1,
    delta_hyper_p=0.9,
    rho_hyper=1.0,
    pi_hyper=1.0,
    alpha_hyper_hyper_mean=100.0,
    alpha_hyper_hyper_scale=1.0,
    alpha_hyper_scale=0.5,
    epsilon_hyper_alpha=1.5,
    epsilon_hyper_beta=1.5 / 0.01,
    mu_hyper_mean=1.0,
    mu_hyper_scale=1.0,
    #     m_hyper_r=1.,
    m_eps=1e-5,
    dtype=torch.float32,
    device="cpu",
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
        #         m_hyper_r,
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
            #             m_hyper_r,
            m_eps,
        ]
    )

    _zero = as_torch(0, dtype=dtype, device=device).squeeze()
    _one = as_torch(1, dtype=dtype, device=device).squeeze()

    # Genotypes
    #     delta_hyper_p = pyro.sample('delta_hyper_p', dist.Beta(1., 1.))
    with pyro.plate("position", g, dim=-1):
        with pyro.plate("strain", s, dim=-2):
            gamma = pyro.sample(
                "gamma", dist.RelaxedBernoulli(temperature=gamma_hyper, logits=_zero)
            )
            # Position presence/absence
            delta = pyro.sample(
                "delta",
                dist.RelaxedBernoulli(
                    temperature=delta_hyper_temp, probs=delta_hyper_p
                ),
            )
    #             delta = pyro.sample(
    #                 'delta', dist.Beta(delta_hyper_p * delta_hyper_temp, (1 - delta_hyper_p) * delta_hyper_temp)
    #             )

    # Meta-community composition
    #     rho_betas = pyro.sample('rho_betas', dist.Beta(_one, rho_hyper).expand([s - 1]).to_event())
    #     rho = pyro.deterministic('rho', stickbreaking_betas_to_probs(rho_betas))
    rho = pyro.sample(
        "rho", dist.Dirichlet(rho_hyper * torch.ones(s, dtype=dtype, device=device))
    )

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

    # Depth at each position
    nu = pyro.deterministic("nu", pi @ delta)
    m_hyper_r = pyro.sample("m_hyper_r", dist.LogNormal(loc=_one, scale=_one))
    m = pyro.sample(
        "m",
        NegativeBinomialReparam(
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
        ).to_event(),
    )


def condition_model(
    model, data=None, device="cpu", dtype=torch.float32, **model_kwargs
):
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
    obs = {k: obs[k].detach().cpu().numpy().squeeze() for k in obs.keys()}
    return obs
