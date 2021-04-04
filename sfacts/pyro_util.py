
import pyro
import pyro.distributions as dist
import torch
from sfacts.logging_util import info


def as_torch(x, dtype=None, device=None):
    # Cast inputs and set device
    return torch.tensor(x, dtype=dtype, device=device)

def all_torch(dtype=None, device=None, **kwargs):
    # Cast inputs and set device
    return {k: as_torch(kwargs[k], dtype=dtype, device=device) for k in kwargs}

def shape_info(model, *args, **kwargs):
    _trace = pyro.poutine.trace(model).get_trace(*args, **kwargs)
    _trace.compute_log_prob()
    info(_trace.format_shapes())
