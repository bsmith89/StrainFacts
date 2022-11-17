# StrainFacts

StrainFacts is a "strain deconvolution" tool, which infers both strain
genotypes and their relative abundance across samples directly from
metagenotype data.

A "metagenotype" for a particular species in a particular sample is an
multidimensional array
counting the number of reads (from shotgun metagenomic sequencing, usually)
that contained a particular allele at
polymorphic sites in that species's genome.
Metagenotypes from multiple (maybe _many_) samples stacked together form
the main input data-type analyzed by StrainFacts.

#### Citation

For detailed information, check out the paper:

> B.J. Smith, X. Li, Z.J. Shi, A. Abate, K.S. Pollard.
Scalable microbial strain inference in metagenomic data using StrainFacts.
_Frontiers in Bioinformatics_ (2022)
doi:[10.3389/fbinf.2022.867386](https://doi.org/10.3389/fbinf.2022.867386)


## Installation

The simplest possible installation directly from GitHub

```
pip install git+https://github.com/bsmith89/StrainFacts.git#egg=sfacts
```

However, installing [as a conda environment](#with-conda) (described below) is
probably the more robust option.

Installing PyTorch/Pyro with GPU support may be more challenging.
Consider using a Docker/Singularity container like
[this one](https://hub.docker.com/r/bsmith89/sfacts_dev), which has many of the
necessary prerequisites (but excluding StrainFacts itself).

## Examples

Short vignettes explaining how to use StrainFacts for data processing,
visualization, and evaluation are provided as Jupyter Notebooks:

- [`examples/simulate_data.ipynb`](https://byronjsmith.com/StrainFacts/simulate_data.html)
- [`examples/filter_data.ipynb`](https://byronjsmith.com/StrainFacts/filter_data.html)
- [`examples/fit_metagenotype.ipynb`](https://byronjsmith.com/StrainFacts/fit_metagenotype.html)
- [`examples/fit_metagenotype_advanced.ipynb`](https://byronjsmith.com/StrainFacts/fit_metagenotype_advanced.html)
- [`examples/evaluate_simulation_fit.ipynb`](https://byronjsmith.com/StrainFacts/evaluate_simulation_fit.html)

## Usage

(**NOTE**: All of the files described in those notebooks and the usage
examples below can be built automatically with Make
e.g. try running: `make examples/sim.filt.fit.world.nc`.)

### Get help

Get CLI help:

```
sfacts --help  # List subcommands
sfacts fit --help  # Subcommand specific CLI usage
```

### Fit a model to metagenotype data

Fit metagenotype data with 15 strains and default everything:

```
sfacts fit \
    --verbose \
    --num-strains 15 \
    --random-seed 0 \
    sim.filt.mgen.nc sim.filt.fit.world.nc
```

For the manuscript, StrainFacts was run with the following hyperparameters set
explicitly:

```
sfacts fit -m ssdd3_with_error  \
    --verbose --device cpu \
    --precision 32 \
    --random-seed 0 \
    --strains-per-sample 0.3 \
    --num-positions 5000 \
    --hyperparameters gamma_hyper=1e-10 \
    --hyperparameters pi_hyper=0.3 \
    --hyperparameters rho_hyper=0.5 \
    --hyperparameters alpha_hyper_mean=10.0 \
    --hyperparameters alpha_hyper_scale=1e-06 \
    --anneal-hyperparameters gamma_hyper=1.0 \
    --anneal-hyperparameters rho_hyper=5.0 \
    --anneal-hyperparameters pi_hyper=1.0 \
    --anneal-steps 10000 --anneal-wait 2000 \
    --optimizer-learning-rate 0.05 \
    --min-optimizer-learning-rate 1e-06 \
    --max-iter 1_000_000 --lag1 50 --lag2 100 \
    examples/sim.filt.mgen.nc examples/sim.filt.fit.world.nc
```

In addition, values for `--strains-per-sample`/`--num-strains`, `--num-positions`, `--precision`, `--random-seed`, `--device` and
input/output files were as described in [the paper](#citation).

### Data Formats

Intermediate files generated
by StrainFacts use a NetCDF-based file format, and harness the
[xarray library](https://docs.xarray.dev/en/stable/) for reading and writing[^why-netcdf].
Such files include input metagenotypes (suggested file extension `*.mgen.nc`),
as well as more complex collections of both latent and observed variables
aligned over their strain/sample/position axes (`*.world.nc`).
All of these files are designed to be loaded and worked with using the `sfacts`
Python library[^netcdf].

[^why-netcdf]: This has benefits for both computational and programmer efficiency.
[^netcdf]: It should be fairly straight-forward to load these NetCDF formatted
files in other programming environments, but this has not been tested.

For users who prefer to interact with StrainFacts via the command-line
interface (CLI) and plain-text files,
scripts are provided as subcommands to convert metagenotype TSVs to this format
(see `sfacts load --metagenotype`).
as well as to export key parameter estimates after fitting (see `sfacts dump`).

For example, to export tab-delimited relative abundance and genotypes tables
from `examples/sim.filt.fit.world.nc`:

```
sfacts dump examples/sim.filt.fit.world.nc \
    --genotype examples/sim.filt.fit.geno.tsv \
    --community examples/sim.filt.fit.comm.tsv
```

```
head sim.filt.fit.refit.geno.tsv sim.filt.fit.refit.comm.tsv

# OUTPUT:
#
# ==> sim.filt.fit.refit.geno.tsv <==
# strain	position	genotypes
# 0	0	0.046489883
# 0	1	0.038462497
# 0	2	9.877303e-07
# 0	3	7.4534853e-07
# 0	4	0.9227885
# 0	5	0.99933404
# 0	6	0.9999933
# 0	7	0.9999994
# 0	8	2.3873392e-06
# 
# ==> sim.filt.fit.refit.comm.tsv <==
# sample	strain	communities
# 0	0	0.00062094553
# 0	1	0.00034649053
# 0	2	0.00017691542
# 0	3	0.00060282537
# 0	4	0.000553946
# 0	5	0.0006001372
# 0	6	0.00032402115
# 0	7	0.6081786
# 0	8	0.00042524032
```

## See Also

### Alternative deconvolution tools:

- [Strain Finder](https://github.com/cssmillie/StrainFinder)
- [ConStrains](https://bitbucket.org/luo-chengwei/constrains/src/master/)
- [MixtureS](http://www.cs.ucf.edu/~xiaoman/mixtureS/)
- [DESMAN](https://github.com/chrisquince/DESMAN)
- [PStrain](https://github.com/wshuai294/PStrain)

Strain Finder is the most similar to StrainFacts, and works well for reasonably
sized data. However, it can be very slow as the number of strains and samples
gets large.

### Metagenotypers

- [GT-Pro](https://github.com/zjshi/gt-pro)
- [MIDAS / IGGtools](https://github.com/czbiohub/iggtools)
- [StrainPhlan](https://github.com/biobakery/metaphlan)

GT-Pro is the preferred metagenotyper for StrainFacts.
While other metagenotypers will also work
(e.g. MIDAS or StrainPhlan assuming their outputs are formatted correctly),
only bi-allelic and core genome SNP sites are currently supported.

### UHGG

GT-Pro is based on the
[UHGG v1.0](http://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v1.0/)
species reference database.
This is a subset of the recently updated 
[MGnify / UHGG v2.0](https://www.ebi.ac.uk/metagenomics/genome-catalogues/human-gut-v2-0)
database.
The MGnify website provides a convenient way to browse this reference.

## How to Hack on StrainFacts

### Editable Installation

First, clone the repository:

```
git clone https://github.com/bsmith89/StrainFacts.git
cd StrainFacts
```

If you want to modify the example notebooks, configure the clean/smudge filter:

`make .git_init`

Then install with either conda or pip:

#### With `conda`

Build the environment:

```
conda env create -n sfacts-dev -f envs/sfacts-dev.yaml
# Also see `make .conda` to run this command automatically.
conda activate sfacts-dev
```

This includes an embedded `pip install --editable` for StrainFacts from the
cloned repo.


#### With `pip`

```
pip install -r requirements.pip
pip install -e .
```

### Integration Testing

```
make clean
make test
```

This runs a complex workflow, including

- simulation
- data conversion
- model fitting
- genotype refitting
- evaluation of estimates against the simulated ground-truth

### Prototyping in Jupyter

```
make start_jupyter
```

## TODO

- [x] A script for converting GT-Pro output TSVs to StrainFacts input data
- [x] A script for extracting estimates from StrainFacts files to TSVs
- [x] Complete installation instructions, as well as a robust artifact for
      running StrainFacts on various platforms (e.g. a Dockerfile or Conda
      environment specification)
- [x] Example data with a tutorial/vignette for running StrainFacts and
      interpreting the output.
    - [x] Simple example fitting a metagenotype
    - [x] Flesh out the example with an explanation of all of the steps.
    - [ ] ~~Example data from SRA with a GT-Pro step, instead of simulated~~
    - [x] Update README/Makefile/Examples to match new CLI interfaces.
- [x] The parameters selected for use in this paper (which we have found to be
      applicable across a range of datasets) will be set as the default.
    - Caveat: All of the hyperparameter annealing parameters should still be set by the user
- [x] We will document some useful approaches for hyperparameter tuning.
- [x] Improve the CLI documentation.
    - [ ] Explicit blurb about how to run the model as in the paper
    - [ ] Model description strings.
- [ ] Refactor for the best looking code
    - [x] De-pluralize core datatypes (community not communities)
    - [x] Remove unused CLI apps/workflows/plotting code
    - [x] Use Python `logging` module instead of shoe-string `info()` function.
    - [ ] Drop stale code in `sfacts.data`, `sfacts.evaluation`, and `sfacts.plot`
    - [ ] Simplify the model structure decorator.
    - [ ] Fix mismatch between 'gamma'/'pi' and 'genotype'/'community'. (Probably just change variable names in the model, yeah?)
    - [x] Move choice of strain numbers for fit and init apps to a set of helper functions.
    - [ ] Refactor CLI app architecture to be more intuitive.
- [ ] Memoization of expensive calculations on wrapped Datasets and DataArrays.
- [ ] Better tools for model selection (e.g. more evaluation scores based on only
      metagenotypes and fits)
- [ ] Split out metagenotype/community/genotype handling to its own package:
      `mgtp` and make this a StrainFacts dependency.
