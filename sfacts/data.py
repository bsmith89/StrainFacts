from sfacts.logging_util import info
from sfacts.pandas_util import idxwhere
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


class WrappedDataArrayMixin():
    constraints = {}
    
    # The following are all white-listed and
    # transparently passed through to self.data, but with
    # different symantics for the return value.
    dims = ()
    safe_unwrapped = ['shape', 'sizes', 'to_pandas', 'min', 'max', 'mean', 'values']
    safe_lifted = ['isel', 'sel']
    variable_name = None
    
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
            raise NotImplementedError("Generic stacking has only been implemented for 2D wrapped DataArrays")
        axis = cls.dims.index(dim)
        data = []
        for k, d in mapping.items():
            if prefix:
                d = d.to_pandas().rename(lambda s: f"{k}_{s}", axis=axis).stack().to_xarray()
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
        elif name in self.safe_lifted:
            return lambda *args, **kwargs: self.__class__(getattr(self.data, name)(*args, **kwargs))
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' "
                                 f"and this name is not found in '{self.__class__.__name__}.dims', "
                                 f"'{self.__class__.__name__}.safe_unwrapped', "
                                 f"or '{self.__class__.__name__}.safe_lifted'. "
                                 f"Consider working with the '{self.__class__.__name__}.data' "
                                 f"xr.DataArray object directly.")
    
    def validate_fast(self):
        assert len(self.data.shape) == len(self.dims)
        assert self.data.dims == self.dims
        
    def validate_constraints(self):
        self.validate_fast()
        for name in self.constraints:
            assert self.constraints[name](self.data), f"Failed constraint: {name}"
        
    def lift(self, func, *args, **kwargs):
        return self.__class__(self.data.pipe(func, *args, **kwargs))
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.data})'
    
    def to_world(self):
        return World(self.data.to_dataset())


class Genotypes(WrappedDataArrayMixin):
    dims = ('strain', 'position')
    constraints = dict(
        on_2_simplex = _on_2_simplex
    )
    variable_name = 'genotypes'
    
    def fuzz_missing(self, missingness, eps=1e-10):
        clip = partial(np.clip, a_min=eps, a_max=(1 - eps))
        return self.lift(lambda g, m: sp.special.expit(sp.special.logit(clip(g)) * clip(m)), m=missingness.data)
    
    # TODO: Move distance metrics to a new module?
    @staticmethod
    def _convert_to_sign_representation(p):
        "Alternative representation of binary genotype on a [-1, 1] interval."
        return p * 2 - 1


    @staticmethod
    def _genotype_sign_representation_dissimilarity(x, y):
        "Dissimilarity between 1D genotypes, accounting for fuzzyness."
        dist = ((x - y) / 2) ** 2
        weight = (x * y) ** 2
        wmean_dist = ((weight * dist).mean()) / ((weight.mean()) + 1)
        return wmean_dist

    @staticmethod
    def _genotype_dissimilarity(x, y):
        return _genotype_s_distance(
            _genotype_p_to_s(x), _genotype_p_to_s(y)
        )
    
    @staticmethod
    def _genotype_dissimilarity_cdmat(unwrapped_values, quiet=True):
        g_sign = Genotypes._convert_to_sign_representation(unwrapped_values)
        s, _ = g_sign.shape
        cdmat = np.empty((s * (s - 1)) // 2)
        k = 0
        with tqdm(total=len(cdmat), disable=quiet) as pbar:
            for i in range(0, s - 1):
                for j in range(i + 1, s):
                    cdmat[k] = Genotypes._genotype_sign_representation_dissimilarity(g_sign[i], g_sign[j])
                    k = k + 1
                    pbar.update()
        return cdmat

    @staticmethod
    def _correlation_distance_cdmat(unwrapped_values, quiet=True):
        return pdist(unwrapped_values, metric='correlation')
    
    def pdist(self, dim='strain', quiet=True):
        index = getattr(self, dim)
        if dim == 'strain':
            unwrapped_values = self.values
            cdmat = self._genotype_dissimilarity_cdmat(unwrapped_values, quiet=quiet)
        elif dim == 'position':
            unwrapped_values = self.values.T
            cdmat = self._correlation_distance_cdmat(unwrapped_values, quiet=quiet)
        # Reboxing
        dmat = pd.DataFrame(squareform(cdmat), index=index, columns=index)
        return dmat
    
    def linkage(self, dim='strain', quiet=True, **kwargs):
        dmat = self.pdist(dim=dim, quiet=quiet)
        cdmat = squareform(dmat)
        kw = dict(method="complete")
        kw.update(kwargs)
        return linkage(cdmat, **kw)
    
    @property
    def entropy(self):
        p = self.data
        q = 1 - p
        ent = -(p * np.log2(p) + q * np.log2(q))
        return ent.sum("position").rename("entropy")


class Missingness(WrappedDataArrayMixin):
    dims = ('strain', 'position')
    constraints = dict(
        on_2_simplex = _on_2_simplex
    )
    variable_name = 'missingness'
        
        
class Communities(WrappedDataArrayMixin):
    dims = ('sample', 'strain')
    constraints = dict(
        strains_sum_to_1 = lambda d: (d.sum('strain') == 1.0).all()
    )
    variable_name = 'communities'


class Overdispersion(WrappedDataArrayMixin):
    dims = ('sample',)
    constraints = dict(
        strains_sum_to_1 = _strictly_positive
    )
    variable_name = 'overdispersion'


class ErrorRate(WrappedDataArrayMixin):
    dims = ('sample',)
    constraints = dict(
        on_2_simplex = _on_2_simplex
    )
    variable_name = 'error_rate'
    

class Metagenotypes(WrappedDataArrayMixin):
    dims = ('sample', 'position', 'allele')
    constraints = dict(
        positive_counts = _positive_counts
    )
    variable_name = 'metagenotypes'

    @classmethod
    def load(cls, filename_or_obj, validate=True):
        data = xr.open_dataarray(filename_or_obj).squeeze().rename({'library_id': 'sample'})
        result = cls(data)
        if validate:
            result.validate_constraints()
        return result
    
    @classmethod
    def from_counts_and_totals(cls, y, m, coords=None):
        if coords is None:
            coords = {}
        if not 'allele' in coords:
            coords['allele'] = ['alt', 'ref']
        x = np.stack([y, m - y], axis=-1)
        return cls.from_ndarray(x, coords=coords)
        
    def dump(self, path, validate=True):
        if validate:
            self.validate_constraints()
        self.data.astype(np.uint8).to_dataset(name="tally").to_netcdf(
            path,
            encoding=dict(tally=dict(zlib=True, complevel=6))
        )

    def select_variable_positions(self, incid_thresh, allele_thresh=0):
        # TODO: Consider using .lift() to do this.
        x = self.data
        minor_allele_incid = (x > allele_thresh).mean("sample").min("allele")
        variable_positions = idxwhere(minor_allele_incid.to_series() > incid_thresh)
        return self.__class__(x.sel(position=variable_positions))

    def select_samples_with_coverage(self, cvrg_thresh):
        # TODO: Consider using .lift() to do this.
        x = self.data
        covered_samples = (x.sum("allele") > 0).mean("position") > cvrg_thresh
        return self.__class__(x.sel(sample=covered_samples))

    def frequencies(self, pseudo=0.):
        "Convert metagenotype counts to a frequencies with optional pseudocount."
        return (self.data + pseudo) / (self.data.sum('allele') + pseudo * self.sizes['allele'])
    
    def dominant_allele_fraction(self, pseudo=0.):
        "Convert metagenotype counts to a frequencies with optional pseudocount."
        return self.frequencies(pseudo=pseudo).max('allele')

    @property
    def genotypes(self):
        return Genotypes(self.frequencies(pseudo=1.).sel(allele='alt').rename({'sample': 'strain'}))

    def to_counts_and_totals(self, binary_allele='alt'):
        return dict(y=self.data.sel(allele=binary_allele).values, m=self.data.sum('allele').values)


class World():
    safe_lifted = ['isel', 'sel']
    safe_unwrapped = []
    dims = ('sample', 'position', 'strain', 'allele')
    variables = [Genotypes, Missingness, Communities, Metagenotypes, ErrorRate, Overdispersion]
    _variable_wrapper_map = {wrapper.variable_name: wrapper for wrapper in variables}
    
    def __init__(self, data):
        self.data = data  # self._align_dims(data)
        self.validate_fast()
        
#     @classmethod
#     def _align_dims(cls, data):
#         missing_dims = [k for k in cls.dims if k not in data.dims]
#         return data.expand_dims(missing_dims).transpose(*cls.dims)
        
    def validate_fast(self):
        assert not (set(self.data.dims) - set(self.dims)), f"Found data dims that shouldn't exist: {self.data.dims}"
        
    def validate_constraints(self):
        self.validate_fast()
        for variable_name in _variable_wrapper_map:
            if variable_name in self.data:
                wrapped_variable = getattr(self, name)
                wrapped_variable.validate_constraints()
    
    @property
    def fuzzed_genotypes(self):
        return self.genotypes.fuzz_missing(self.missingness)
    
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
            return lambda *args, **kwargs: self.__class__(getattr(self.data, name)(*args, **kwargs))
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}' "
                                 f"and this name is not found in '{self.__class__.__name__}.dims', "
                                 f"'{self.__class__.__name__}.safe_unwrapped', "
                                 f"or '{self.__class__.__name__}.safe_lifted'. "
                                 f"Consider working with the '{self.__class__.__name__}.data' object directly.")
