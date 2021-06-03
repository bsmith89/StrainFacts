import sfacts as sf
import pyro
import pyro.distributions as dist
import torch
from torch.nn.functional import pad as torch_pad


SHARED_DIMS = ("sample", "position", "strain", "allele")
SHARED_DESCRIPTIONS = dict(
    gamma=("strain", "position"),
    delta=("strain", "position"),
    rho=("strain",),
    pi=("sample", "strain"),
    epsilon=("sample",),
    m_hyper_r=("sample",),
    m_hyper_r_mean=(),
    m_hyper_r_scale=(),
    mu=("sample",),
    nu=("sample", "position"),
    p_noerr=("sample", "position"),
    p=("sample", "position"),
    alpha_hyper_mean=(),
    alpha=("sample",),
    m=("sample", "position"),
    y=("sample", "position"),
    genotypes=("strain", "position"),
    missingness=("strain", "position"),
    communities=("sample", "strain"),
    metagenotypes=("sample", "position", "allele"),
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
    log_p_raised = log_p / alpha
    log_q_raised = log_q / beta
    return torch.exp(
        log_p_raised - torch.logsumexp(torch.stack([log_p_raised, log_q_raised]), dim=0)
    )
