import sfacts as sf
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import TorchDistribution
from torch.distributions import constraints
from torch.nn.functional import pad as torch_pad

# from sfacts.pyro_util import log1mexp
import warnings

# import scipy as sp
# import numpy as np


SHARED_DIMS = ("sample", "position", "strain", "allele")
SHARED_DESCRIPTIONS = dict(
    gamma=("strain", "position"),
    delta=("strain", "position"),
    rho=("strain",),
    pi_hyper_sample=("sample",),
    pi=("sample", "strain"),
    epsilon=("sample",),
    epsilon_hyper=(),
    m_hyper_r=("sample",),
    m_hyper_r_mean=(),
    m_hyper_r_scale=(),
    mu_hyper_mean=(),
    mu=("sample",),
    nu=("sample", "position"),
    p_noerr=("sample", "position"),
    p=("sample", "position"),
    alpha_hyper_mean=(),
    alpha=("sample",),
    m=("sample", "position"),
    y=("sample", "position"),
    genotype=("strain", "position"),
    missingness=("strain", "position"),
    community=("sample", "strain"),
    metagenotype=("sample", "position", "allele"),
)


def _mapping_subset(mapping, keys):
    return {k: mapping[k] for k in keys}


def stickbreaking_betas_to_probs(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return torch_pad(beta, (0, 1), value=1) * torch_pad(beta1m_cumprod, (1, 0), value=1)


def stickbreaking_betas_to_log_probs(beta):
    log_beta = torch.log(beta)
    # log_beta1m_cumprod = log1mexp(log_beta).cumsum(-1)
    log_beta1m_cumprod = torch.log((1 - beta)).cumsum(-1)
    return torch_pad(log_beta, (0, 1), value=0) + torch_pad(
        log_beta1m_cumprod, (1, 0), value=0
    )


def NegativeBinomialReparam(mu, r, validate_args=True):
    p = 1.0 / ((r / mu) + 1.0)
    logits = torch.logit(p)
    #     p = torch.clamp(p, eps, 1 - eps)
    return dist.NegativeBinomial(
        total_count=r,
        logits=logits,
        validate_args=validate_args,
    )


def unit_interval_power_transformation(p, alpha, beta, eps=0.0):
    log_p = torch.log(p)
    log_q = torch.log1p(-p)
    log_p_raised = log_p / alpha
    log_q_raised = log_q / beta
    result = torch.exp(
        log_p_raised - torch.logsumexp(torch.stack([log_p_raised, log_q_raised]), dim=0)
    )
    return (result + eps) / (1 + 2 * eps)


def k_simplex_power_transformation1(p, alpha, eps=0.0):
    p_raised = p ** (1 / alpha) + eps
    return p_raised / p_raised.sum(dim=-1, keepdims=True)


def k_simplex_power_transformation(p, alpha, eps=0.0):
    kp1 = p.shape[-1]
    log_p = torch.log(p)
    log_p_raised = log_p / alpha
    result = torch.exp(
        log_p_raised - torch.logsumexp(log_p_raised, dim=-1, keepdims=True)
    )
    return (result + eps) / (1 + kp1 * eps)


def powerperturb_transformation(p, power, perturb):
    log_p = torch.log(p)
    log_perturb = torch.log(perturb)
    log_y_unnorm = (power * log_p) + log_perturb
    return torch.exp(log_y_unnorm - torch.logsumexp(log_y_unnorm, -1, keepdim=True))


def powerperturb_transformation_inverse(p, power, perturb):
    power_rev = 1 / power
    perturb_rev = 1 / powerperturb_transformation(
        perturb, power=power_rev, perturb=torch.tensor(1)
    )
    return powerperturb_transformation(p, power=power_rev, perturb=power_rev)


def powerperturb_transformation_unit_interval(p, power, perturb):
    p = torch.stack([p, 1 - p], dim=-1)
    return powerperturb_transformation(p, power, perturb)[..., 0]


def shifted_scaled_dirichlet_loglik(alpha, p, a, x):
    D = x.shape[-1]
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    termA = torch.lgamma(sum_alpha) - torch.sum(
        torch.lgamma(alpha), dim=-1, keepdim=True
    )
    termB = -(D - 1) * torch.log(a)
    termC_num = torch.sum(
        (-(alpha / a) * torch.log(p)) + (alpha / a - 1) * torch.log(x),
        dim=-1,
        keepdim=True,
    )
    termC_den = sum_alpha * torch.log(
        torch.sum((x / p) ** (1 / a), dim=-1, keepdim=True)
    )
    return termA + termB + termC_num - termC_den


class ShiftedScaledDirichlet(TorchDistribution):
    support = pyro.distributions.Dirichlet.support
    has_rsample = False
    arg_constraints = {
        "alpha": constraints.positive,
        "p": constraints.unit_interval,
        "a": constraints.positive,
    }

    def __init__(self, alpha, p, a, validate_args=None):
        alpha, p, a = torch.distributions.utils.broadcast_all(
            alpha, p, a.unsqueeze(dim=-1)
        )
        a = a[..., [0]]
        self._dirichlet = pyro.distributions.Dirichlet(concentration=alpha)
        self.p = p
        self.a = a
        super(TorchDistribution, self).__init__(
            self._dirichlet.batch_shape,
            self._dirichlet.event_shape,
            validate_args=validate_args,
        )

    @property
    def alpha(self):
        return self._dirichlet.concentration

    def sample(self, sample_shape=torch.Size()):
        y = self._dirichlet.sample(sample_shape)
        return sf.model_zoo.components.powerperturb_transformation(y, self.a, self.p)

    def log_prob(self, value):
        return shifted_scaled_dirichlet_loglik(
            self.alpha, self.p, self.a, value
        ).squeeze(dim=-1)


def logit_laplace_loglik(loc, scale, x):
    return (
        -torch.log(scale)
        - torch.log(2 * x)
        - torch.log1p(-x)
        - torch.abs(torch.log(x) - torch.log1p(-x) - loc) / scale
    )


class LogitLaplace(TorchDistribution):
    support = pyro.distributions.Beta.support
    has_rsample = False
    arg_constraints = {
        "mu": constraints.real,
        "b": constraints.positive,
    }

    def __init__(self, mu, b, validate_args=None):
        mu, b = torch.distributions.utils.broadcast_all(
            mu,
            b,
        )
        self._laplace = pyro.distributions.Laplace(loc=mu, scale=b)
        super(TorchDistribution, self).__init__(
            self._laplace.batch_shape,
            self._laplace.event_shape,
            validate_args=validate_args,
        )

    @property
    def mu(self):
        return self._laplace.loc

    @property
    def b(self):
        return self._laplace.scale

    def sample(self, sample_shape=torch.Size()):
        y = self._laplace.sample(sample_shape)
        return torch.special.expit(y)

    def log_prob(self, value):
        return logit_laplace_loglik(
            self.mu,
            self.b,
            value,
        )


class LogTriangle(TorchDistribution):
    support = constraints.unit_interval
    has_rsample = False
    arg_constraints = {
        "a": constraints.real,
    }

    def __init__(self, a, validate_args=None):
        # FIXME: I don't understand how to set up the shapes and pass them to super()
        a, low, high = torch.distributions.utils.broadcast_all(
            a, torch.zeros_like(a), torch.ones_like(a)
        )
        self.a = a
        self._uniform = dist.Uniform(low, high)
        super().__init__(self._uniform._batch_shape, validate_args=validate_args)

    @property
    def _normalizer(self):
        a = self.a
        return 2 * (torch.exp(a / 2) - 1) / a

    def _half_prob(self, value):
        a, normalizer = self.a, self._normalizer
        return torch.exp(a * value) / normalizer

    def _prob(self, value):
        first_half = self._half_prob(value)
        second_half = self._half_prob(1 - value)
        return torch.where(value < 0.5, first_half, second_half)

    def log_prob(self, value):
        return torch.log(self._prob(value))

    def _half_inv_cdf(self, q):
        a, normalizer = self.a, self._normalizer
        return torch.log(q * normalizer * a + 1) / a

    def _inv_cdf(self, q):
        first_half = self._half_inv_cdf(q)
        second_half = 1 - self._half_inv_cdf(1 - q)
        return torch.where(q < 0.5, first_half, second_half)

    def sample(self, sample_shape=()):
        u = self._uniform.sample(sample_shape)
        return self._inv_cdf(u)


class LogSoftTriangle(TorchDistribution):
    support = constraints.unit_interval
    has_rsample = False
    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }

    def __init__(self, a, b, validate_args=None):
        a, b = torch.distributions.utils.broadcast_all(a, b)
        low, high = torch.zeros_like(a), torch.ones_like(a)
        self.a, self.b = a, b
        self._uniform = dist.Uniform(low, high)
        super().__init__(self._uniform._batch_shape, validate_args=validate_args)

    @property
    def _normalizer(self):
        a, b = self.a, self.b
        return (torch.exp(a) - 1) / a + (torch.exp(b) - 1) / b

    def log_prob(self, value):
        a, b, c = self.a, self.b, self._normalizer
        x = value
        return torch.log((torch.exp(a * x) + torch.exp(b * (1 - x))) / c)

    def cdf(self, value):
        a, b = self.a, self.b
        exp = torch.exp
        x = value
        return (b * (-exp(a * x)) + a * exp(b) * (exp(-b * x) - 1) + b) / (
            a * (-exp(b)) - exp(a) * b + a + b
        )

    # def icdf(self, value):
    #     out = np.array([sp.optimize.bisect(lambda x: (v - self.cdf(x)).cpu().numpy(), 0, 1) for v in value.flatten()])
    #     breakpoint()
    #     return torch.tensor(out).reshape(*value.shape)
    #
    def sample(self, sample_shape=()):
        _approx = LogTriangle(a=self.a)
        warnings.warn(
            "Using LogTriangle as an approximation for random sampling. This is probably a bad idea."
        )
        return _approx.sample(sample_shape=sample_shape)
