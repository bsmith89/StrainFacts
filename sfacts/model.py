import pyro
import pyro.distributions as dist
import torch
from functools import partial
from torch.nn.functional import pad as torch_pad
import xarray as xr
import sfacts as sf
from sfacts.pyro_util import all_torch
from sfacts.logging_util import info


class Structure:
    def __init__(self, generative, dims, description, default_hyperparameters=None):
        """

        *generative* :: Pyro generative model function(shape_dim_0, shape_dim_1, shape_dim_2, ..., **hyper_parameters)
        *dims* :: Sequence of names for dim_0, dim_1, dim_2, ...
        *description* :: Mapping from model variable to its dims.
        *default_hyperparameters* :: Values to use for hyperparameters when not explicitly set.
        """
        if default_hyperparameters is None:
            default_hyperparameters = {}

        self.generative = generative
        self.dims = dims
        self.description = description
        self.default_hyperparameters = default_hyperparameters

    #         _ = self(self._dummy_shape, **all_torch(**self.default_hyperparameters))

    #         info(f"New Structure({self.generative}, {self.default_hyperparameters})")

    def __call__(self, shape, data, hyperparameters):
        assert len(shape) == len(self.dims)
        conditioned_generative = pyro.condition(self.generative, data)
        return conditioned_generative(*shape, **hyperparameters)

    #     def condition(self, **data):
    #         new_data = self.data.copy()
    #         new_data.update(data)
    #         return self.__class__(
    #             generative=self.generative,
    #             dims=self.dims,
    #             description=self.description,
    #             default_hyperparameters=self.default_hyperparameters,
    #             data=new_data,
    #         )

    @property
    def _dummy_shape(self):
        shape = range(1, len(self.dims) + 1)
        return shape

    def explain_shapes(self, shape=None):
        if shape is None:
            shape = self._dummy_shape
        info(dict(zip(self.dims, shape)))
        sf.pyro_util.shape_info(self(shape, **self.default_hyperparameters))

    def __repr__(self):
        return (
            f"{self.generative.__name__}("
            # f"{self.generative}, "
            f"dims={self.dims}, "
            f"description={self.description}, "
            f"default_hyperparameters={self.default_hyperparameters} "
            f")"
        )


# For decorator use.
def structure(dims, description, default_hyperparameters=None):
    return partial(
        Structure,
        dims=dims,
        description=description,
        default_hyperparameters=default_hyperparameters,
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
    ):
        if hyperparameters is None:
            hyperparameters = {}

        if data is None:
            data = {}

        self.structure = structure
        self.coords = {k: self._coords_or_range(coords[k]) for k in self.structure.dims}
        self.dtype = dtype
        self.device = device
        self.hyperparameters = self.structure.default_hyperparameters.copy()
        self.hyperparameters.update(hyperparameters)
        self.data = data

    @property
    def sizes(self):
        return {k: len(self.coords[k]) for k in self.structure.dims}

    @property
    def shape(self):
        return tuple(self.sizes.values())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.structure}, "
            f"coords={self.coords}, "
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"hyperparameters={self.hyperparameters}, "
            f"data={self.data})"
        )

    def __call__(self):
        # Here's where all the action happens.
        # All parameters are cast based on dtype and device.
        # The model is conditioned on the
        # data, and then called with the shape tuple
        # and cast hyperparameters.
        return self.structure(
            self.shape,
            data=all_torch(**self.data, dtype=self.dtype, device=self.device),
            hyperparameters=all_torch(
                **self.hyperparameters, dtype=self.dtype, device=self.device
            ),
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
        )

    def format_world(self, data):
        out = {}
        for k in self.structure.description:
            out[k] = xr.DataArray(
                data[k],
                dims=self.structure.description[k],
                coords={dim: self.coords[dim] for dim in self.structure.description[k]},
            )
        return sf.data.World(xr.Dataset(out))

    def simulate(self, n=1, seed=None):
        sf.pyro_util.set_random_seed(seed)
        obs = pyro.infer.Predictive(self, num_samples=n)()
        obs = {k: obs[k].detach().cpu().numpy().squeeze() for k in obs.keys()}
        return obs

    def simulate_world(self, seed=None):
        return self.format_world(self.simulate(n=1))
