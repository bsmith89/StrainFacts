import sfacts as sf
import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import logging
from functools import cached_property
from warnings import warn


class Error(Exception):
    pass


class DataConstraintError(Error):
    pass


class DataDimensionsError(Error):
    pass


class DataConstraint:
    def __init__(self, name, test_func, allow_empty=True):
        self.name = name
        self.test_func = test_func
        self.allow_empty = allow_empty

    def __call__(self, data):
        if data.to_series().empty:
            if self.allow_empty:
                logging.warn(f"{self.name} not tested because data was empty: {data}")
                return True
            else:
                return False
        else:
            return self.test_func(data)

    def raise_error(self, data):
        raise DataConstraintError(f"Failed constraint: {self.name}")


ON_2_SIMPLEX = DataConstraint(
    "on_2_simplex",
    lambda d: (d.min() >= 0) and (d.max() <= 1.0),
)
STRICTLY_POSITIVE = DataConstraint(
    "strictly_positive",
    lambda d: d.min() > 0,
)
POSITIVE_COUNTS = DataConstraint(
    "positive_counts",
    lambda d: (d.astype(int) == d).all() and (d.min() >= 0),
)


class WrappedDataArrayMixin:
    constraints = {}

    # The following are all white-listed and
    # transparently passed through to self.data, but with
    # different symantics for the return value.
    dims = ()
    safe_lifted = [
        "isel",
        "sel",
        "drop_sel",
        "drop_isel",
    ]
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
        "reindex",
    ]
    # safe_lifted = []
    variable_name = None

    @classmethod
    def load(cls, filename_or_obj, validate=True):
        world = World.load(filename_or_obj=filename_or_obj, validate=validate)
        return getattr(world, cls.variable_name)

    def dump(self, path, validate=True, **kwargs):
        self.to_world().dump(path=path, validate=validate, **kwargs)

    @classmethod
    def _post_load(cls, data, validate=True):
        result = cls(data)
        if validate:
            result.validate_constraints()
        return result

    @classmethod
    def load_from_tsv(cls, path, validate=True):
        data = pd.read_table(path, index_col=cls.dims)
        data = data.squeeze().to_xarray()
        data.name = cls.variable_name
        return cls._post_load(data)

    def dump_to_tsv(self, path, validate=True):
        self.validate_constraints()
        self.data.to_series().to_csv(path, sep="\t")

    @classmethod
    def from_ndarray(cls, x, coords=None):
        if coords is None:
            coords = {k: None for k in cls.dims}
        shapes = {k: x.shape[i] for i, k in enumerate(cls.dims)}
        for k in coords:
            if coords[k] is None:
                coords[k] = range(shapes[k])
        data = xr.DataArray(
            x,
            dims=cls.dims,
            coords=coords,
        )
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
        if name == self.variable_name:
            return self
        elif name in self.dims:
            return getattr(self.data, name)
        elif name in self.safe_unwrapped:
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
                f"Consider working with the '{self.__class__.__name__}.data' "
                f"xr.DataArray object directly."
            )

    def validate_fast(self):
        if not (
            (len(self.data.shape) == len(self.dims)) and (self.data.dims == self.dims)
        ):
            raise DataDimensionsError(self.data.dims, self.dims)

    def validate_constraints(self):
        self.validate_fast()
        for constraint in self.constraints:
            if not constraint(self.data):
                constraint.raise_error(self.data)

    def lift(self, func, *args, **kwargs):
        return self.__class__(func(self.data, *args, **kwargs))

    def mlift(self, name, *args, **kwargs):
        func = getattr(self.data, name)
        return self.__class__(func(*args, **kwargs))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    @classmethod
    def concat(cls, data, dim, rename=True):
        out_data = []
        new_coords = []
        for name in data:
            d = data[name].data
            out_data.append(d)
            if rename:
                prefix = f"{name}_"
            else:
                prefix = ""
            new_coords.extend([f"{prefix}{i}" for i in d[dim].values])
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


class Metagenotype(WrappedDataArrayMixin):
    """Counts of alleles across samples and positions."""

    dims = ("sample", "position", "allele")
    constraints = [POSITIVE_COUNTS]
    variable_name = "metagenotype"

    @classmethod
    def load_from_tsv(cls, path, validate=True):
        data = pd.read_table(path, index_col=cls.dims)
        data = data.squeeze().to_xarray()
        data = data.fillna(0)  # This is required for metagenotypes, specifically.
        data.name = cls.variable_name
        return cls._post_load(data)

    @classmethod
    def load_from_merged_gtpro(cls, path, validate=True, **kwargs):
        data = pd.read_table(
            path,
            names=[
                "sample_id",
                "species_id",
                "global_pos",
                "contig",
                "local_pos",
                "ref_allele",
                "alt_allele",
                "ref_count",
                "alt_count",
            ],
            **kwargs,
        ).rename(
            columns={
                "sample_id": "sample",
                "global_pos": "position",
                "ref_count": "ref",
                "alt_count": "alt",
            }
        )
        assert len(data.species_id.unique()) == 1
        data = (
            data[["sample", "position", "ref", "alt"]]
            .set_index(["sample", "position"])
            .rename_axis(columns="allele")
            .stack()
            .squeeze()
        )
        data = (
            data.astype(int).reorder_levels(cls.dims).sort_index().to_xarray().fillna(0)
        )
        return cls._post_load(data)

    @classmethod
    def from_counts_and_totals(cls, y, m, coords=None):
        if coords is None:
            coords = {}
        if not "allele" in coords:
            coords["allele"] = ["alt", "ref"]
        x = np.stack([y, m - y], axis=-1)
        return cls.from_ndarray(x, coords=coords)

    def to_csv(self, *args, **kwargs):
        self.data.to_series().to_csv(*args, **kwargs)

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
        return self.mlift("sel", sample=(self.horizontal_coverage() >= cvrg_thresh))

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

    def to_estimated_genotype(self, pseudo=0.0, fillna=True):
        frac = (
            self.alt_allele_fraction(pseudo=pseudo)
            .rename({"sample": "strain"})
            .reset_coords(drop=True)
        )
        if fillna:
            frac = frac.fillna(0.5)
        return Genotype(frac)

    def total_counts(self):
        return self.data.sum("allele")

    def allele_counts(self, allele="alt"):
        return self.data.sel(allele=allele)

    def mean_depth(self, dim="sample"):
        if dim == "sample":
            over = "position"
        elif dim == "position":
            over = "sample"
        return self.total_counts().mean(over)

    def horizontal_coverage(self, min_count=1, dim="sample"):
        if dim == "sample":
            over = "position"
        elif dim == "position":
            over = "sample"
        return (self.total_counts() >= min_count).mean(over)

    def to_counts_and_totals(self, binary_allele="alt"):
        return dict(
            y=self.allele_counts(allele=binary_allele).values,
            m=self.total_counts().values,
        )

    @cached_property
    def _sample_pdist_pseudo0(self):
        return (
            self.to_estimated_genotype(pseudo=0.0)
            .pdist(dim="strain")
            .rename_axis(columns="sample", index="sample")
        )

    def pdist(self, dim="sample", pseudo=0.0, **kwargs):
        # Use the special cached property for this common invocation.
        if (dim == "sample") and (pseudo == 0.0) and (not kwargs):
            return self._sample_pdist_pseudo0
        if dim == "sample":
            _dim = "strain"
        else:
            _dim = dim
        return (
            self.to_estimated_genotype(pseudo=pseudo)
            .pdist(dim=_dim, **kwargs)
            .rename_axis(columns=dim, index=dim)
        )

    def cosine_pdist(self, dim="sample"):
        if dim != "sample":
            raise NotImplementedError("Only dim 'sample' has been implemented.")
        d = self.to_series().unstack(dim).T
        return pd.DataFrame(
            squareform(pdist(d.values, metric="cosine")), index=d.index, columns=d.index
        )

    def podlesny_pdist(self, dim="sample"):
        assert dim == "sample", "Not Implemented"
        d = self.data.values
        return pd.DataFrame(
            sf.math.podlesny_cdist(d, d), index=self.sample, columns=self.sample
        )

    def clusters(self, s_or_thresh, linkage="complete", **kwargs):
        if s_or_thresh < 1:
            s = None
            thresh = float(s_or_thresh)
        elif s_or_thresh > 1:
            s = int(s_or_thresh)
            thresh = None
        dist = self.pdist("sample", **kwargs)
        return pd.Series(
            AgglomerativeClustering(
                n_clusters=s,
                distance_threshold=thresh,
                affinity="precomputed",
                linkage=linkage,
            ).fit_predict(dist),
            index=dist.columns,
        )

    def linkage(self, dim="sample", pseudo=0.0, **kwargs):
        if dim == "sample":
            _dim = "strain"
        else:
            _dim = dim
        return self.to_estimated_genotype(pseudo=pseudo).linkage(dim=_dim, **kwargs)

    def cosine_linkage(
        self,
        dim="sample",
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.cosine_pdist(dim=dim)
        cdmat = squareform(dmat)
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def podlesny_linkage(
        self,
        dim="sample",
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.podlesny_pdist(dim=dim)
        cdmat = squareform(dmat)
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def entropy(self, dim="sample", norm=1):
        if dim == "sample":
            over = "position"
        elif dim == "position":
            over = "sample"

        ent_scaled = (
            sf.math.entropy(self.frequencies(), "allele") ** norm
        ) * self.total_counts()
        return (
            (ent_scaled.sum(over) / self.total_counts().sum(over)) ** (1 / norm)
        ).rename("entropy")


class Genotype(WrappedDataArrayMixin):
    dims = ("strain", "position")
    constraints = [ON_2_SIMPLEX]
    variable_name = "genotype"

    @classmethod
    def load_from_tsv(cls, path, validate=True):
        data = (
            pd.read_table(path, index_col=["strain", "position", "allele"])
            .xs("alt", level="allele")
            .squeeze()
            .to_xarray()
        )
        data.name = cls.variable_name
        return cls._post_load(data)

    def discretized(self):
        return self.lift(np.round)

    def fuzzed(self, eps=1e-5):
        return self.lift(lambda x: (x + eps) / (1 + 2 * eps))

    def cdist(self, other, **kwargs):
        "Compare distances between strain genotypes."
        # Gather indexes
        self_strains = self.strain
        other_strains = other.strain
        positions = list(set(self.position.values) & set(other.position.values))

        # Align data
        self = self.sel(position=positions).values
        other = other.sel(position=positions).values

        # Calculate distances
        dmat = sf.math.genotype_cdist(self, other, **kwargs)
        # Reboxing
        dmat = pd.DataFrame(dmat, index=self_strains, columns=other_strains)
        return dmat

    def pdist(self, dim="strain", **kwargs):
        index = getattr(self, dim)
        if dim == "strain":
            unwrapped_values = self.values
            _kwargs = dict()
            _kwargs.update(kwargs)
            cdmat = sf.math.genotype_pdist(unwrapped_values, **_kwargs)
        elif dim == "position":
            unwrapped_values = self.values.T
            _kwargs = dict(metric="cosine")
            _kwargs.update(kwargs)
            cdmat = pdist(sf.math.genotype_binary_to_sign(self.values.T), **_kwargs)
        # Reboxing
        dmat = pd.DataFrame(squareform(cdmat), index=index, columns=index)
        return dmat

    def linkage(
        self,
        dim="strain",
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.pdist(dim=dim)
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
        d = sf.math.genotype_binary_to_sign(d)
        cdmat = pdist(d, metric="cosine")
        return pd.DataFrame(squareform(cdmat), index=index, columns=index)

    def cosine_linkage(
        self, dim="strain", method="complete", optimal_ordering=False, **kwargs
    ):
        cdmat = squareform(self.cosine_pdist(dim=dim))
        return linkage(
            cdmat, method=method, optimal_ordering=optimal_ordering, **kwargs
        )

    def entropy(self, dim="strain", norm=1):
        if dim == "strain":
            over = "position"
        elif dim == "position":
            over = "strain"
        p = self.data
        ent = (
            pd.DataFrame(
                sf.math.binary_entropy(p), columns=self.position, index=self.strain
            )
            .rename_axis(columns="position", index="strain")
            .stack()
            .to_xarray()
        )
        return ((ent ** norm).mean(over) ** (1 / norm)).rename("entropy")

    def clusters(self, thresh, linkage="complete", **kwargs):
        dist = self.pdist("strain", **kwargs)
        return pd.Series(
            AgglomerativeClustering(
                distance_threshold=thresh,
                n_clusters=None,
                affinity="precomputed",
                linkage=linkage,
            ).fit_predict(dist),
            index=dist.columns,
        )


class Community(WrappedDataArrayMixin):
    dims = ("sample", "strain")
    constraints = [
        DataConstraint(
            "strains_sum_to_1",
            lambda d: np.allclose(d.sum("strain"), 1.0, atol=1e-4),
        )
    ]
    variable_name = "community"

    def fuzzed(self, eps=1e-5):
        new_data = self.data + eps
        new_data = new_data / new_data.sum("strain")
        return self.__class__(new_data)

    def pdist(self, dim="sample"):
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
        dim="sample",
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.pdist(dim=dim)
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
        ent = sf.math.entropy(p, axis=sum_over)
        return pd.Series(ent, index=getattr(p, dim)).rename_axis(index=dim).to_xarray()

    def renormalize(self):
        return Community(
            self.data.to_series()
            .unstack()
            .apply(lambda x: x / x.sum(), axis=1)
            .stack()
            .to_xarray()
        )


class World:
    safe_lifted = ["isel", "sel", "drop_sel"]
    safe_unwrapped = ["sizes"]
    dims = ("sample", "position", "strain", "allele")
    variables = [Genotype, Community, Metagenotype]
    _variable_wrapper_map = {wrapper.variable_name: wrapper for wrapper in variables}

    def __init__(self, data):
        self.data = data  # self._align_dims(data)
        self.validate_fast()

    #     @classmethod
    #     def _align_dims(cls, data):
    #         missing_dims = [k for k in cls.dims if k not in data.dims]
    #         return data.expand_dims(missing_dims).transpose(*cls.dims)

    @cached_property
    def metagenotype(self):
        name = "metagenotype"
        return self._variable_wrapper_map[name](self.data[name])

    @cached_property
    def genotype(self):
        name = "genotype"
        return self._variable_wrapper_map[name](self.data[name])

    @cached_property
    def community(self):
        name = "community"
        return self._variable_wrapper_map[name](self.data[name])

    @classmethod
    def from_combined(cls, *args):
        return cls(xr.Dataset({v.variable_name: v.data for v in args}))

    def validate_fast(self):
        if set(self.data.dims) - set(self.dims):
            raise DataDimensionsError(self.data.dims, self.dims)

    def validate_constraints(self):
        self.validate_fast()
        for variable_name in self._variable_wrapper_map:
            if variable_name in self.data:
                wrapped_variable = getattr(self, variable_name)
                wrapped_variable.validate_constraints()

    def cull_empty_dims(self):
        empty_dims = []
        for dim in self.data.dims:
            if self.sizes[dim] == 0:
                empty_dims.append(dim)
        warn(f"Empty dimensions {empty_dims} removed from dataset.")
        return World(self.data.drop_dims(empty_dims))

    def dump(self, path, validate=True, **kwargs):
        if validate:
            self.validate_constraints()
        self.data.to_netcdf(path, engine="netcdf4", **kwargs)

    @classmethod
    def load(cls, filename_or_obj, validate=True):
        data = xr.load_dataset(filename_or_obj, engine="netcdf4")
        world = cls(data)
        if validate:
            world.validate_constraints()
        return world

    @classmethod
    def open(cls, filename_or_obj, validate=True):
        data = xr.open_dataset(filename_or_obj)
        world = cls(data)
        if validate:
            world.validate_constraints()
        return world

    @classmethod
    def peek_netcdf_sizes(cls, filename_or_obj):
        data = xr.open_dataset(filename_or_obj)
        sizes = data.sizes
        data.close()
        return sizes

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

    def __getattr__(self, name):
        if name in self.dims:
            # Return dims for those registered in self.dims.
            return getattr(self.data, name)
        elif name in self._variable_wrapper_map:
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

    @cached_property
    def _unifrac_pdist_discretized(self):
        return sf.unifrac.unifrac_pdist(self, discretized=True)

    def unifrac_pdist(self, discretized=True, **kwargs):
        if discretized and (not kwargs):
            return self._unifrac_pdist_discretized
        return sf.unifrac.unifrac_pdist(self, discretized=discretized, **kwargs)

    def unifrac_linkage(
        self,
        method="complete",
        optimal_ordering=False,
        **kwargs,
    ):
        dmat = self.unifrac_pdist(**kwargs)
        cdmat = squareform(dmat)
        return linkage(cdmat, method=method, optimal_ordering=optimal_ordering)

    def merge_strains(self, relabel, discretized=False):
        clust = relabel
        total_strain_depth = (
            (self.metagenotype.mean_depth("sample") * self.community.data)
            .sum("sample")
            .to_series()
        )

        def weighted_mean_genotype(genotype, total_depth, **kwargs):
            try:
                out = np.average(genotype, weights=total_depth, **kwargs)
            except ZeroDivisionError:
                out = np.average(genotype, **kwargs)
            return out

        genotype = Genotype(
            self.genotype.to_series()
            .unstack("strain")
            .groupby(clust, axis="columns")
            .apply(
                lambda x: pd.Series(
                    weighted_mean_genotype(
                        x, total_strain_depth.loc[x.columns], axis=1
                    ),
                    index=x.index,
                )
            )
            .rename_axis(columns="strain")
            .T.stack()
            .to_xarray()
        )
        community = Community(
            self.community.to_series()
            .unstack("strain")
            .groupby(clust, axis="columns")
            .sum()
            .rename_axis(columns="strain")
            .stack()
            .to_xarray()
        ).renormalize()
        if "metagenotype" in self.data:
            world = World.from_combined(genotype, community, self.metagenotype)
        else:
            world = World.from_combined(genotype, community)
        return world

    def drop_low_abundance_strains(self, thresh):
        is_abundant = (self.community.max("sample") >= thresh).to_series()
        relabel = self.strain.to_series().where(is_abundant, -1)
        return self.merge_strains(relabel, discretized=False)

    def drop_high_entropy_strains(self, thresh, norm=1):
        is_low_entropy = (self.genotype.entropy(norm=norm) <= thresh).to_series()
        relabel = self.strain.to_series().where(is_low_entropy, -1)
        return self.merge_strains(relabel, discretized=False)

    def collapse_similar_strains(self, thresh, discretized=False, **kwargs):
        if discretized:
            geno = self.genotype.discretized()
        else:
            geno = self.genotype
        relabel = geno.clusters(thresh=thresh, **kwargs)
        return self.merge_strains(relabel, discretized=discretized)

    def reassign_high_community_entropy_samples(self, thresh, norm=1):
        high_entropy_samples = (self.community.entropy(norm=norm) > thresh).to_series()
        comm = self.community.to_series().unstack("strain")
        comm.loc[:, -1] = 0
        comm.loc[high_entropy_samples, :] = 0
        comm.loc[high_entropy_samples, -1] = 1
        comm = sf.Community(comm.stack().to_xarray()).renormalize()
        geno = self.genotype.to_series().unstack()
        geno.loc[-1, :] = 0.5
        geno = sf.Genotype(geno.stack().to_xarray())
        return sf.World.from_combined(comm, geno, self.metagenotype)

    def reassign_plurality_strain(self):
        comm = sf.Community(
            self.community.to_series()
            .unstack("strain")
            .apply(lambda x: x == x.max(), axis=1)
            .astype(float)
            .stack()
            .to_xarray()
        )
        return sf.World.from_combined(comm, self.genotype, self.metagenotype)

    def expected_dominant_allele_fraction(self):
        return (
            (self.community.data @ self.genotype.data)
            .to_series()
            .to_frame(name="alt")
            .rename_axis(columns="allele")
            .assign(ref=lambda x: 1 - x)
            .stack()
            .to_xarray()
            .transpose(*Metagenotype.dims)
            .max("allele")
        )


def latent_metagenotype_pdist(world, dim="sample"):
    if dim == "sample":
        dim = "strain"
    return Genotype(world.data.p.rename({"sample": "strain"})).pdist(dim=dim)


def latent_metagenotype_linkage(
    world, dim="sample", method="complete", optimal_ordering=False
):
    return linkage(
        squareform(latent_metagenotype_pdist(world, dim=dim)),
        method=method,
        optimal_ordering=optimal_ordering,
    )
