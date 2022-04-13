# StrainFacts

StrainFacts is a "strain deconvolution" tool, which infers both strain
genotypes and their relative abundance across samples directly from
metagenotype data[^metagenotype-meaning].

[^metagenotype-meaning]: A "metagenotype" is a data matrix of counts of how
many reads in a shotgun metagenomic sequence library contained each allele at a
set of polymorphic sites.
Metagenotypes from multiple (maybe _many_) samples stacked together form
the main input data-type analyzed by StrainFacts.

For detailed information, check out the manuscript:

> Scalable microbial strain inference in metagenomic data using StrainFacts.
B.J. Smith, X. Li, A. Abate, Z.J. Shi, K.S. Pollard
_bioRxiv_ doi:[10.1101/2022.02.01.478746](https://doi.org/10.1101/2022.02.01.478746)


## TODO

- [x] A script for converting GT-Pro output TSVs to StrainFacts input data
- [x] A script for extracting estimates from StrainFacts files to TSVs
- [ ] Complete installation instructions, as well as a robust artifact for
      running StrainFacts on various platforms (e.g. a Dockerfile or Conda
      environment specification)
- [ ] Example data with a tutorial/vignette for running StrainFacts and
      interpreting the output.
    - [ ] Simple example fitting a metagenotype
    - [ ] Put example data on Zenodo so that users don't need to build it themselves.
    - [ ] Flesh out the example with an explanation of all of the steps.
    - [ ] Example data from SRA with a GT-Pro step, instead of simulated
- [x] The parameters selected for use in this paper (which we have found to be
      applicable across a range of datasets) will be set as the default.
    - Caveat: All of the hyperparameter annealing parameters should still be set by the user
- [ ] We will document some useful approaches for hyperparameter tuning.
- [ ] Improve the CLI documentation.
- [ ] Remove unused CLI apps/workflows/plotting code
- [ ] Refactor for the best looking code
    - [ ] De-pluralize core datatypes (community not communities)

## Installation

### With Pip

1. Install pre-requisite python packages[^pyro-install]

```
pip install -r requirements.pip
```

[^pyro-install]: Installing PyTorch/Pyro with GPU support may be more challenging.
Consider using a Docker/Singularity container like
[this one](https://hub.docker.com/repository/docker/bsmith89/compbio).

2. Install StrainFacts:

```
pip install git+https://github.com/bsmith89/StrainFacts.git#egg=sfacts
```

### For development

Clone the repository:

```
git clone https://github.com/bsmith89/StrainFacts.git
cd StrainFacts
```

Initialize the conda environment:

```
make .conda
# runs: conda env create -n sfacts-dev -f envs/sfacts-dev.yaml
```

This includes `pip install --editable` for StrainFacts from the clones repo.

## Usage

Get CLI usage information:

```
sfacts --help  # List subcommands
sfacts fit --help  # Subcommand specific CLI usage
```

Fit metagenotype data with default everything:

```
sfacts fit --num-strains <S> examples/example0.sim.filt.mgen.tsv examples/example0.sim.filt.fit.nc
```

For the manuscript, StrainFacts was run with the following hyperparameters:

```
sfacts fit -m ssdd3_with_error  \
    --verbose --device cpu \
    --precision 32 \
    --random-seed 0 \
    --strains-per-sample 0.3 \
    --num-positions 5000 \
    --nmf-init \
    --hyperparameters gamma_hyper=1e-10 \
    --hyperparameters pi_hyper=0.3 \
    --hyperparameters rho_hyper=0.5 \
    --hyperparameters alpha_hyper_mean=10.0 \
    --hyperparameters alpha_hyper_scale=1e-06 \
    --anneal-hyperparameters gamma_hyper=1.0 \
    --anneal-hyperparameters rho_hyper=5.0 \
    --anneal-hyperparameters pi_hyper=1.0 \
    --anneal-hyperparameters alpha_hyper_mean=10.0 \
    --anneal-steps 10000 --anneal-wait 2000 \
    --optimizer-learning-rate 0.05 \
    --min-optimizer-learning-rate 1e-06 \
    --max-iter 1_000_000 --lag1 50 --lag2 100 \
    --tsv \
    examples/example0.sim.filt.mgen.tsv examples/example0.sim.filt.fit.nc
    ```

(If the file `examples/example0.sim.mgen.tsv` is missing, it can be reconstructed
by running `make examples/example0.sim.mgen.tsv`.)

## Data Formats

### Inputs

TODO

### Outputs

For both computational and programmer efficiency, intermediate files generated
by StrainFacts use the NetCDF format, and harness the xarray library for
reading and writing.
Such files include input metagenotypes (suggested file extension `*.mgen.nc`),
as well as simulations (`*.sim.nc`) with both latent and observed variables.
All of these are easy to read and work with using the `sfacts` library.

For users who prefer to interact with StrainFacts via the command-line
interface (CLI), scripts are provided as subcommands to convert TSVs to these
formats (see `sfacts load_mgen`).
as well as to extract key inferences (see `sfacts dump`).

#### Example

Export text relative abundance and genotypes tables from `examples/example0.sim.filt.fit.nc`:

```
sfacts dump examples/example0.sim.filt.fit.nc --tsv --genotype examples/example0.sim.filt.fit.geno.tsv --community examples/example0.sim.filt.fit.comm.tsv
```

## See Also

### Alternative deconvolution tools:

- [Strain Finder](TODO)
- [ConStrains](TODO)
- [MixtureS](TODO)
- [DESMAN](TODO)
- [PStrain](TODO)

Strain Finder is the most similar to StrainFacts, and works well for reasonably
sized data. However, it can be very slow as the number of strains and samples
gets large.

### Metagenotypers

- [GT-Pro](https://github.com/zjshi/gt-pro)
- [MIDAS / IGGtools](https://github.com/czbiohub/iggtools)
- [StrainPhlan](TODO)

GT-Pro is the preferred metagenotyper for Strain Finder.
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
