import pyro
import torch
import sfacts as sf
import logging


PRECISION_MAP = {32: torch.float32, 64: torch.float64}


def as_torch(x, dtype=None, device=None):
    # Cast inputs and set device
    if isinstance(x, torch.Tensor):
        return torch.tensor(x.numpy(), dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)


def all_torch(dtype=None, device=None, **kwargs):
    # Cast inputs and set device
    return {k: as_torch(kwargs[k], dtype=dtype, device=device) for k in kwargs}


def shape_info(model, *args, **kwargs):
    _trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
    _trace.compute_log_prob()
    return _trace.format_shapes()


def set_random_seed(seed):
    if seed is None:
        logging.warn("No RNG seed set. Results will not be deterministic.")
    else:
        pyro.set_rng_seed(seed)


def log1mexp(x):
    x = torch.abs(x)
    return torch.where(
        x < torch.log(torch.tensor(2.0)),
        torch.log(torch.expm1(-x)),
        torch.log1p(-torch.exp(-x)),
    )
