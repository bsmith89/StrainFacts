import pyro
import torch
from functools import partial
import xarray as xr
import sfacts as sf
from sfacts.pyro_util import all_torch
from warnings import warn
from pprint import pformat
from collections import UserDict
import logging


class Hyperparameters(UserDict):
    """Static list of hyperparameter names and their mutable values."""

    def __init__(self, **kwargs):
        self.static_keys = list(kwargs.keys())
        super(Hyperparameters, self).__init__(**kwargs)

    def in_keys(self, key):
        return key in self.static_keys

    def __iter__(self):
        return iter(self.static_keys)

    def __setitem__(self, key, item):
        if self.in_keys(key):
            self.data[key] = item
        else:
            warn(f"Hyperparameter '{key}' not in existing keys. Value ignored.")

    def update(self, other):
        for k in other:
            self[k] = other[k]

    def pformat(self):
        return self.__class__.__name__ + "(**" + pformat(self.data) + ")"


class Structure:
    def __init__(
        self, generative, dims, description, default_hyperparameters, text_summary
    ):
        """

        *generative* :: Pyro generative model function(shape_dim_0, shape_dim_1, shape_dim_2, ..., **hyper_parameters)
        *dims* :: Sequence of names for dim_0, dim_1, dim_2, ...
        *description* :: Mapping from model variable to its dims.
        *default_hyperparameters* :: Values to use for hyperparameters when not explicitly set.
        """
        self.generative = generative
        self.dims = dims
        self.description = description
        self.default_hyperparameters = default_hyperparameters
        self.text_summary = text_summary

    def __call__(self, shape, data, hyperparameters, unit):
        assert len(shape) == len(self.dims)
        conditioned_generative = pyro.condition(self.generative, data)
        return conditioned_generative(*shape, **hyperparameters, _unit=unit)

    @property
    def _dummy_shape(self):
        shape = range(1, len(self.dims) + 1)
        return shape

    # def explain_shapes(self, shape=None):
    #     if shape is None:
    #         shape = self._dummy_shape
    #     print(dict(zip(self.dims, shape)))
    #     print(sf.pyro_util.shape_info(self(shape, self.default_hyperparameters)))

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ("generative=" + repr(self.generative.__qualname__) + ", ")
            + ("dims=" + repr(self.dims) + ", ")
            + ("description=" + repr(self.description) + ", ")
            + ("default_hyperparameters=" + repr(self.default_hyperparameters) + ", ")
            + ")"
        )

    def pformat(self, indent=1):
        return (
            self.__class__.__name__
            + "("
            + "\n"
            + " " * indent
            + " generative="
            + self.generative.__qualname__
            + ",\n"
            + " " * indent
            + " dims="
            + pformat(self.dims, indent=indent + 1)
            + ",\n"
            + " " * indent
            + " description="
            + pformat(self.description, indent=indent + 1)
            + ",\n"
            + " " * indent
            + " default_hyperparameters="
            + self.default_hyperparameters.pformat(indent=indent + 1)
            + "\n"
            + " " * (indent - 1)
            + ")"
        )


# For decorator use.
def structure(dims, description, default_hyperparameters, text_summary=""):
    return partial(
        Structure,
        dims=dims,
        description=description,
        default_hyperparameters=Hyperparameters(**default_hyperparameters),
        text_summary=text_summary,
    )


class ParameterizedModel:
    def __init__(
        self,
        structure,
        coords,
        dtype=torch.float32,
        device="cpu",
        data=None,
        hyperparameters=None,
        passed_hyperparameters=None,
    ):
        if hyperparameters is None:
            hyperparameters = {}

        if passed_hyperparameters is None:
            passed_hyperparameters = []

        if data is None:
            data = {}

        # Special case of alleles because they are format in different
        # ways for genotypes (0, 1) and metagenotypes (alt-count + total_count).
        if "allele" in coords:
            if "alt" in coords["allele"]:
                if list(coords["allele"]).index("alt") > 0:
                    warn(
                        "Weird things can happen if binary (alt/ref) allele coordinates are passed as ['ref', 'alt'] (instead of ['alt', 'ref'])."
                    )

        self.structure = structure
        self.coords = {k: self._coords_or_range(coords[k]) for k in self.structure.dims}
        self.dtype = dtype
        self.device = device
        self.hyperparameters = self.structure.default_hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.data = data
        self.passed_hyperparameters = passed_hyperparameters

    @property
    def sizes(self):
        return {k: len(self.coords[k]) for k in self.structure.dims}

    @property
    def shape(self):
        return tuple(self.sizes.values())

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + ("structure=" + repr(self.structure) + ", ")
            + ("coords=" + repr(self.coords) + ", ")
            + ("dtype=" + repr(self.dtype) + ", ")
            + ("device=" + repr(self.device) + ", ")
            + ("hyperparameters=" + repr(self.hyperparameters) + ", ")
            + ("data=" + repr(self.data) + ", ")
            + ("passed_hyperparameters=" + repr(self.passed_hyperparameters) + ", ")
            + ")"
        )

    def pformat(self, indent=1):
        return (
            self.__class__.__name__
            + "("
            + "\n"
            + " " * indent
            + "structure="
            + self.structure.pformat(indent=indent + 1)
            + ",\n"
            + " " * indent
            + "coords="
            + pformat(self.coords, indent=indent + 1)
            + ",\n"
            + " " * indent
            + "dtype="
            + pformat(self.dtype, indent=indent + 1)
            + ",\n"
            + " " * indent
            + "device="
            + pformat(self.dtype, indent=indent + 1)
            + ",\n"
            + " " * indent
            + "hyperparameters="
            + self.hyperparameters.pformat(indent=indent + 1)
            + ",\n"
            + " " * indent
            + "data="
            + pformat(self.data, indent=indent + 1)
            + "\n"
            + " " * (indent - 1)
            + "passed_hyperparameters="
            + pformat(self.passed_hyperparameters, indent=indent + 1)
            + "\n"
            + " " * (indent - 1)
            + ")"
        )

    def __call__(self, *args):
        # Here's where all the action happens.
        # All parameters are cast based on dtype and device.
        # The model is conditioned on the
        # data, and then called with the shape tuple
        # and cast hyperparameters.
        data = all_torch(**self.data, dtype=self.dtype, device=self.device)
        hyperparameters = all_torch(
            **self.hyperparameters, dtype=self.dtype, device=self.device
        )
        hyperparameters.update(
            {
                self.passed_hyperparameters[i]: args[i]
                for i in range(len(self.passed_hyperparameters))
            }
        )
        return self.structure(
            self.shape,
            data=data,
            hyperparameters=hyperparameters,
            unit=torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )

    @staticmethod
    def _coords_or_range(coords):
        if type(coords) == int:
            return range(coords)
        else:
            return coords

    def with_hyperparameters(self, **hyperparameters):
        new_hyperparameters = self.hyperparameters.copy()
        new_hyperparameters.update(hyperparameters)
        return self.__class__(
            structure=self.structure,
            coords=self.coords,
            dtype=self.dtype,
            device=self.device,
            hyperparameters=new_hyperparameters,
            data=self.data,
            passed_hyperparameters=self.passed_hyperparameters,
        )

    def with_amended_coords(self, **coords):
        new_coords = self.coords.copy()
        new_coords.update(coords)
        return self.__class__(
            structure=self.structure,
            coords=new_coords,
            dtype=self.dtype,
            device=self.device,
            hyperparameters=self.hyperparameters,
            data=self.data,
            passed_hyperparameters=self.passed_hyperparameters,
        )

    def condition(self, **data):
        new_data = self.data.copy()
        new_data.update(data)
        return self.__class__(
            structure=self.structure,
            coords=self.coords,
            dtype=self.dtype,
            device=self.device,
            hyperparameters=self.hyperparameters,
            data=new_data,
            passed_hyperparameters=self.passed_hyperparameters,
        )

    def with_passed_hyperparameters(self, *args, replace=True):
        if not replace:
            new_passed_hyperparameters = self.passed_hyperparameters.copy()
            new_passed_hyperparameters.extend(args)
        else:
            new_passed_hyperparameters = args
        return self.__class__(
            structure=self.structure,
            coords=self.coords,
            dtype=self.dtype,
            device=self.device,
            hyperparameters=self.hyperparameters,
            data=self.data,
            passed_hyperparameters=new_passed_hyperparameters,
        )

    def format_world(self, data):
        out = {}
        for k in self.structure.description:
            out[k] = xr.DataArray(
                data[k],
                dims=self.structure.description[k],
                coords={dim: self.coords[dim] for dim in self.structure.description[k]},
            )
        return sf.data.World(
            xr.Dataset(
                out,
                attrs=self.hyperparameters,
            )
        )

    def simulate(self, *args, n=1, seed=None):
        sf.pyro_util.set_random_seed(seed)
        obs = pyro.infer.Predictive(self, num_samples=n)(*args)
        obs = {k: obs[k].detach().cpu().numpy().squeeze() for k in obs.keys()}
        return obs

    def simulate_world(self, *args, seed=None):
        return self.format_world(self.simulate(*args, n=1, seed=seed))
