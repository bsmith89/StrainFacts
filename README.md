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


## Installation

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


## Usage

Get CLI usage information:

```
sfacts --help  # List subcommands
sfacts fit --help  # Subcommand specific CLI usage
```

Fit metagenotype data with default everything:

```
sfacts fit --num-strains <S> input_metagenotype.nc output_fit.nc
```

Fit metagenotype data with the same parameters used in the manuscript:

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
    --tsv-input \
    input_metagenotype.tsv \
    output_fit.nc
```

TODO: Implement `--tsv-input` flag.

Where `input_metagenotype.tsv` looks something like:

```
sample	position	allele	tally
sample1	1	alt	1
sample1	2	ref	1
sample1	2	alt	1
sample2	2	ref	3
sample2	2	alt	1
[...continued...]
```

(NOTE: The "sample" and "position" IDs can be anything, but "allele" should be `alt` or `ref`.)

TODO: Add an `example/` directory with e.g. `input_metagenotype.tsv`

Export text relative abundance and genotypes tables from `output_fit.nc`:

```
sfacts export output_fit.nc community.tsv genotype.tsv
```

TODO: Implement this subcommand.

## See Also

### Metagenotypers
- [GT-Pro](https://github.com/zjshi/gt-pro)
- [MIDAS / IGGtools](https://github.com/czbiohub/iggtools)

### Species reference databases

- [UHGG v1.0](http://ftp.ebi.ac.uk/pub/databases/metagenomics/mgnify_genomes/human-gut/v1.0/)
- [MGnify / UHGG v2.0](https://www.ebi.ac.uk/metagenomics/genome-catalogues/human-gut-v2-0)
