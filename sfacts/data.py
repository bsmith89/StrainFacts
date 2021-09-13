from sfacts.math import binary_entropy, entropy, genotype_pdist, genotype_binary_to_sign
import xarray as xr
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
import scipy as sp
from functools import partial


def _on_2_simplex(d):
    return (d.min() >= 0) and (d.max() <= 1.0)


def _strictly_positive(d):
    return d.min() > 0


def _positive_counts(d):
    return (d.astype(int) == d).all()


class WrappedDataArrayMixin:
    constraints = {}

    # The following are all white-listed and
    # transparently passed through to self.data, but with
    # different symantics for the return value.
    dims = ()
    safe_unwrapped = [
        "shape",
        "sizes",
        "to_pandas",
        "to_dataframe",
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "values",
        "pipe",
        "to_series",
        "isel",
        "sel",
        "reindex",
        "drop_sel",
        "drop_isel",
    ]
    # safe_lifted = []
    variable_name = None

    @classmethod
    def from_ndarray(cls, x, coords=None):
        if coords is None:
            coords = {k: None for k in cls.dims}
        shapes = {k: x.shape[i] for i, k in enumerate(cls.dims)}
        for k in coords:
            if coords[k] is None:
                coords[k] = range(shapes[k])
        data = xr.DataArray(x, dims=cls.dims, coords=coords,)
        return cls(data)

    @classmethod
    def stack(cls, mapping, dim, prefix=False, validate=True):
        if not len(cls.dims) == 2:
            raise NotImplementedError(
                "Generic stacking has only been implemented for 2D wrapped DataArrays"
            )
        axis = cls.dims.index(dim)
        data = []
        for k, d in mapping.items():
            if prefix:
                d = (
                    d.to_pandas()
                    .rename(lambda s: f"{k}_{s}", axis=axis)
                    .stack()
                    .to_xarray()
                )
            else:
                d = d.data
            data.append(d)
        out = cls(xr.concat(data, dim=dim))
        if validate:
            out.validate_constraints()
        return out

    def __init__(self, data):
        self.data = data
        self.validate_fast()

    def __getattr__(self, name):
        if name in self.dims:
            return getattr(self.data, name)
        elif name in self.safe_unwrapped:
            return getattr(self.data, name)
        # elif name in self.safe_lifted:
        #     return lambda *args, **kwargs: self.__class__(
        #         getattr(self.data, name)(*args, **kwargs)
        #     )
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"and this name is not found in '{self.__class__.__name__}.dims', "
                f"'{self.__class__.__name__}.safe_unwrapped', "
                f"or '{self.__class__.__name__}.safe_lifted'. "
                f"Consider working with the '{self.__class__.__name__}.data' "
                f"xr.DataArray object directly."
            )

    def validate_fast(self):
        assert len(self.data.shape) == len(self.dims)
        assert self.data.dims == self.dims

    def validate_constraints(self):
        self.validate_fast()
        for name in self.constraints:
            assert self.constraints[name](self.data), f"Failed constraint: {name}"

    def lift(self, func, *args, **kwargs):
        return self.__class__(func(self.data, *args, **kwargs))

    def mlift(self, name, *args, **kwargs):
        func = getattr(self, name)
        return self.__class__(func(*args, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    @classmethod
    def concat(cls, data, dim):
        out_data = []
        new_coords = []
        for name in data:
            d = data[name].data
            out_data.append(d)
            new_coords.extend([f"{name}_{i}" for i in d[dim].values])
        out_data = xr.concat(out_data, dim)
        out_data[dim] = new_coords
        return cls(out_data)

    def to_world(self):
        return World(self.data.to_dataset(name=self.variable_name))

    def random_sample(self, replace=False, keep_order=True, **kwargs):
        isel = {}
        for dim in kwargs:
            n = kwargs[dim]
            dim_n = self.data.sizes[dim]
            ii = np.random.choice(np.arange(dim_n), size=n, replace=replace)
            if keep_order:
                ii = sorted(ii)
            isel[dim] = ii
        return self.__class__(data=self.data.isel(**isel))


class Metagenotypes(WrappedDataArrayMixin):
    """Counts of alleles across samples and positions.

    """

    dims = ("sample", "position", "allele")
    constraints = dict(positive_counts=_positive_counts)
    variable_name = "metagenotypes"

    @classmethod
    def load(cls, filename_or_obj, validate=True):
        data = (
            xr.open_dataarray(filename_or_obj)
            .rename({"library_id": "sample"})
            .squeeze(drop=True)
        )
        data.name = "metagenotypes"
        result = cls(data)
        if validate:
            result.validate_constraints()
        return result

    @classmethod
    def from_counts_and_totals(cls, y, m, coords=None):
        if coords is None:
            coords = {}
        if not "allele" in coords:
            coords["allele"] = ["alt", "ref"]
        x = np.stack([y, m - y], axis=-1)
        return cls.from_ndarray(x, coords=coords)

    def dump(self, path, validate=True):
        if validate:
            self.validate_constraints()
        self.data.astype(np.uint8).to_dataset(name="tally").to_netcdf(
            path, encoding=dict(tally=dict(zlib=True, complevel=6))
        )

    def allele_incidence(self, thresh=1):
        allele_presence = self.data >= thresh
        return allele_presence.sum("sample") / allele_presence.any("allele").sum(
            "sample"
        )

    def select_variable_positions(self, thresh, count_thresh=1):
        return self.mlift(
            "sel",
            position=(
                self.allele_incidence(thresh=count_thresh).min("allele") >= thresh
            ),
        )

    def select_samples_with_coverage(self, cvrg_thresh):
        # TODO: Consider using .lift() to do this.
        x = self.data
        covered_samples = (x.sum("allele") > 0).mean("position") > cvrg_thresh
        return self.mlift("sel", sample=covered_samples)

    def frequencies(self, pseudo=0.0):
        "Convert metagenotype counts to a frequency with optional pseudocount."
        return (self.data + pseudo) / (
            self.data.sum("allele") + pseudo * self.sizes["allele"]
        )

    def dominant_allele_fraction(self, pseudo=0.0):
        "Convert metagenotype counts to a frequencies with optional pseudocount."
        return self.frequencies(pseudo=pseudo).max("allele")

    def alt_allele_fraction(self, pseudo=0.0):
        return self.frequencies(pseudo=pseudo).sel(allele="alt")

    def to_estimated_genotypes(self, pseudo=1.0):
        return Genotypes(
            self.alt_allele_fraction(pseudo=pseudo).rename({"sample": "strain"})
        )

    def total_counts(self):
        return self.data.sum("allele")

    def allele_counts(self, allele="alt"):
        return self.sel(allele=allele)

    def mean_depth(self, dim="sample"):
        if dim == "sample":
            over = "position"
        elif dim == "position":
            over = "sample"
        return self.total_counts().mean(over)

    def to_counts_and_totals(self, binary_allele="alt"):
        return dict(
            y=self.allele_counts(allele=binary_allele).values,
            m=self.total_counts().values,
        )

    def pdist(self, dim="sample", gpseudo=1.0, pseudo=0.0, **kwargs):
        if dim == "sample":
            _dim = "strain"
        else:
            _dim = dim
        return (
            self.to_estimated_genotypes(pseudo=gpseudo)
            .pdist(dim=_dim, pseudo=pseudo, **kwargs)
            .rename_axis(columns=dim, index=dim)
        )

    def cosine_pdist(self, dim="sample"):
        if dim != "sample":
            raise NotImplementedError("Only dim 'sample' has been implemented.")
        d = self.to_series().unstack(dim).T
        return pd.DataFrame(
            squareform(pdist(d.values, metric="cosine")), index=d.index, columns=d.index
        )

    def linkage(self, dim="sample", gpseudo=1.0, pseudo=0.0, **kwargs):
        if dim == "sample":
            _dim = "strain"
        else:
            _dim = dim
        return self.to_estimated_genotypes(pseudo=gpseudo).linkage(
            dim=_dim, pseudo=pseudo, **kwargs
        )

    def cosine_linkage(
        self, dim="sample", method="complete", optimal_ordering=False, **kwargs,
    ):
        dmat = self.cosine_pdist(dim=dim)
        cdmat = squareform(dmat)
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def entropy(self, dim="sample"):
        if dim == "sample":
            over = "position"
        elif dim == "position":
            over = "sample"
        p = self.dominant_allele_fraction()
        ent = binary_entropy(p)
        return ent.sum(over).rename("entropy")


class Genotypes(WrappedDataArrayMixin):
    dims = ("strain", "position")
    constraints = dict(on_2_simplex=_on_2_simplex)
    variable_name = "genotypes"

    def softmask_missing(self, missingness, eps=1e-10):
        clip = partial(np.clip, a_min=eps, a_max=(1 - eps))
        return self.lift(
            lambda g, m: sp.special.expit(sp.special.logit(clip(g)) * clip(m)),
            m=missingness.data,
        )

    def discretized(self):
        return self.lift(np.round)

    def fuzzed(self, eps=1e-5):
        return self.lift(lambda x: (x + eps) / (1 + 2 * eps))


    def pdist(self, dim="strain", pseudo=0.0, quiet=True):
        index = getattr(self, dim)
        if dim == "strain":
            unwrapped_values = self.values
            cdmat = genotype_pdist(unwrapped_values, pseudo=pseudo, quiet=quiet)
        elif dim == "position":
            if pseudo != 0.0:
                raise ValueError(
                    "Position dissimilarity does not involve a 'pseudo' parameter."
                )
            unwrapped_values = self.values.T
            cdmat = pdist(genotype_binary_to_sign(self.values.T), metric="cosine")
        # Reboxing
        dmat = pd.DataFrame(squareform(cdmat), index=index, columns=index)
        return dmat

    def linkage(
        self,
        dim="strain",
        pseudo=0.0,
        quiet=True,
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.pdist(dim=dim, pseudo=pseudo, quiet=quiet)
        cdmat = squareform(dmat)
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def cosine_pdist(self, dim="strain"):
        if dim == "strain":
            d = self.values
            index = self.strain
        elif dim == "position":
            d = self.values.T
            index = self.position
        d = genotype_binary_to_sign(d)
        cdmat = pdist(d, metric="cosine")
        return pd.DataFrame(squareform(cdmat), index=index, columns=index)

    def cosine_linkage(
        self, dim="strain", method="complete", optimal_ordering=False, **kwargs
    ):
        cdmat = squareform(self.cosine_pdist(dim=dim))
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def entropy(self, dim="strain"):
        if dim == "strain":
            sum_over = "position"
        elif dim == "position":
            sum_over = "strain"
        p = self.data
        ent = binary_entropy(p)
        return ent.sum(sum_over).rename("entropy")


class Missingness(WrappedDataArrayMixin):
    dims = ("strain", "position")
    constraints = dict(on_2_simplex=_on_2_simplex)
    variable_name = "missingness"


class Communities(WrappedDataArrayMixin):
    dims = ("sample", "strain")
    constraints = dict(
        strains_sum_to_1=lambda d: np.allclose(d.sum("strain"), 1.0, atol=1e-5)
    )
    variable_name = "communities"

    def fuzzed(self, eps=1e-5):
        new_data = self.data + eps
        new_data = new_data / new_data.sum("strain")
        return self.__class__(new_data)

    def pdist(self, dim="strain", quiet=True):
        index = getattr(self, dim)
        if dim == "strain":
            unwrapped_values = self.values.T
            cdmat = pdist(unwrapped_values, metric="cosine")
        elif dim == "sample":
            unwrapped_values = self.values
            cdmat = pdist(unwrapped_values, metric="braycurtis")
        # Reboxing
        dmat = pd.DataFrame(squareform(cdmat), index=index, columns=index)
        return dmat

    def linkage(
        self,
        dim="strain",
        quiet=True,
        method="average",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.pdist(dim=dim, quiet=quiet)
        cdmat = squareform(dmat)
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def entropy(self, dim="sample"):
        if dim == "strain":
            sum_over = "sample"
        elif dim == "sample":
            sum_over = "strain"
        p = self.data
        ent = entropy(p, axis=sum_over)
        return ent.rename("entropy")


class Overdispersion(WrappedDataArrayMixin):
    dims = ("sample",)
    constraints = dict(strains_sum_to_1=_strictly_positive)
    variable_name = "overdispersion"


class ErrorRate(WrappedDataArrayMixin):
    dims = ("sample",)
    constraints = dict(on_2_simplex=_on_2_simplex)
    variable_name = "error_rate"


class World:
    safe_lifted = ["isel", "sel"]
    safe_unwrapped = ["sizes"]
    dims = ("sample", "position", "strain", "allele")
    variables = [Genotypes, Missingness, Communities, Metagenotypes]
    _variable_wrapper_map = {wrapper.variable_name: wrapper for wrapper in variables}

    def __init__(self, data):
        self.data = data  # self._align_dims(data)
        self.validate_fast()

    #     @classmethod
    #     def _align_dims(cls, data):
    #         missing_dims = [k for k in cls.dims if k not in data.dims]
    #         return data.expand_dims(missing_dims).transpose(*cls.dims)

    def validate_fast(self):
        assert not (
            set(self.data.dims) - set(self.dims)
        ), f"Found data dims that shouldn't exist: {self.data.dims}"

    def validate_constraints(self):
        self.validate_fast()
        for variable_name in self._variable_wrapper_map:
            if variable_name in self.data:
                wrapped_variable = getattr(self, variable_name)
                wrapped_variable.validate_constraints()

    def dump(self, path, validate=True):
        if validate:
            self.validate_constraints()
        self.data.to_netcdf(path)

    @classmethod
    def load(cls, path, validate=True):
        data = xr.open_dataset(path)
        world = cls(data)
        if validate:
            world.validate_constraints()
        return world

    def random_sample(self, replace=False, keep_order=True, **kwargs):
        isel = {}
        for dim in kwargs:
            n = kwargs[dim]
            dim_n = self.data.sizes[dim]
            ii = np.random.choice(np.arange(dim_n), size=n, replace=replace)
            if keep_order:
                ii = sorted(ii)
            isel[dim] = ii
        return self.__class__(data=self.data.isel(**isel))

    @property
    def masked_genotypes(self):
        return self.genotypes.softmask_missing(self.missingness)

    def __getattr__(self, name):
        if name in self.dims:
            # Return dims for those registered in self.dims.
            return getattr(self.data, name)
        if name in self._variable_wrapper_map:
            # Return wrapped variables for those registered in self.variables.
            return self._variable_wrapper_map[name](self.data[name])
        elif name in self.safe_unwrapped:
            # Return a naked version of the variables registered in self.safe_unwrapped
            return getattr(self.data, name)
        elif name in self.safe_lifted:
            # Return a lifted version of the the attributes registered in safe_lifted
            return lambda *args, **kwargs: self.__class__(
                getattr(self.data, name)(*args, **kwargs)
            )
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}' "
                f"and this name is not found in '{self.__class__.__name__}.dims', "
                f"'{self.__class__.__name__}.safe_unwrapped', "
                f"or '{self.__class__.__name__}.safe_lifted'. "
                f"Consider working with the '{self.__class__.__name__}.data' object directly."
            )

    @classmethod
    def concat(cls, data, dim, rename_coords=False):
        new_coords = []
        # Add source metadata and rename concatenation coordinates
        renamed_data = []
        shared_variables = set([str(v) for v in list(data.values())[0].data.variables])
        for name in data:
            d = data[name].data.copy()
            d["_concat_from"] = xr.DataArray(name, dims=(dim,), coords={dim: d[dim]})
            if rename_coords:
                new_coords.extend([f"{name}_{i}" for i in d[dim].values])
            else:
                new_coords.extend(d[dim].values)
            shared_variables &= set([str(v) for v in d.variables])
            renamed_data.append(d)
        # Drop unshared variables
        ready_data = []
        for d in renamed_data:
            ready_data.append(d[list(shared_variables - set(cls.dims))])
        # Concatenate
        out_data = xr.concat(
            ready_data, dim, data_vars="minimal", coords="minimal", compat="override"
        )
        out_data[dim] = new_coords
        return cls(out_data)


def latent_metagenotypes_pdist(world, dim="sample"):
    if dim == "sample":
        dim = "strain"
    return Genotypes(world.data.p.rename({"sample": "strain"})).pdist(dim=dim)


def latent_metagenotypes_linkage(
    world, dim="sample", method="average", optimal_ordering=False
):
    return linkage(
        squareform(latent_metagenotypes_pdist(world, dim=dim)),
        method=method,
        optimal_ordering=optimal_ordering,
    )
