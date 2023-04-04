import argparse
import sys
import warnings
import sfacts as sf
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
from sfacts.app.components import (
    add_hyperparameters_cli_argument,
    parse_hyperparameter_strings,
    add_model_structure_cli_argument,
    parse_model_structure_string,
    add_optimization_arguments,
    transform_optimization_parameter_inputs,
    AppInterface,
    ArgumentMutualExclusionError,
    add_strain_count_cli_arguments,
    transform_strain_count_parameter_inputs,
    calculate_strain_count,
)


class ArgumentConstraintError(Exception):
    pass


class NoOp(AppInterface):
    app_name = "do_nothing"
    description = "Do nothing (dummy subcommand)."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--dummy", type=int, default=1, help="Test option; does nothing."
        )
        add_hyperparameters_cli_argument(parser)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        with sf.logging_util.phase_info("Dummy phase"):
            pass


class Load(AppInterface):
    app_name = "load"
    description = "Build StrainFacts files from input data."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--gtpro-metagenotype",
            help="Path of a metagenotype in 'merged' GT-Pro format; this is the same as output from `GT_Pro parse`, but (1) filtered to just one species, (2) with a sample_id column prepended, and (3) concatenated from N samples.",
        )
        parser.add_argument(
            "--metagenotype",
            help="Path of a metagenotype matrix in standard, StrainFacts TSV format; this is the same as the output of `sfacts dump --metagenotype`.",
        )
        parser.add_argument(
            "--community",
            help="Path of a community matrix in standard, StrainFacts TSV format; this is the same as the output of `sfacts dump --community`.",
        )
        parser.add_argument(
            "--genotype",
            help="Path of a genotype matrix in standard, StrainFacts TSV format; this is the same as the output of `sfacts dump --genotype`.",
        )
        parser.add_argument(
            "outpath", help="Path to write the StrainFacts/NetCDF file."
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        if args.gtpro_metagenotype and args.metagenotype:
            raise ArgumentMutualExclusionError(
                "Only one of --metagenotype or --gtpro-metagenotype may be passed.",
            )
        return args

    @classmethod
    def run(cls, args):
        values = []
        # Only one of gtpro_metagenotype and metagenotype is allowed.
        if args.gtpro_metagenotype:
            values.append(
                sf.data.Metagenotype.load_from_merged_gtpro(args.gtpro_metagenotype)
            )
        if args.metagenotype:
            values.append(sf.data.Metagenotype.load_from_tsv(args.metagenotype))

        if args.community:
            values.append(sf.data.Community.load_from_tsv(args.community))
        if args.genotype:
            values.append(sf.data.Genotype.load_from_tsv(args.genotype))
        world = sf.data.World.from_combined(*values)
        world.dump(args.outpath)


class StackMetagenotypes(AppInterface):
    app_name = "stack_mgen"
    description = "Combine samples from two or more metagenotype files."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "outpath",
            help="Path StrainFacts/NetCDF output.",
        )
        parser.add_argument(
            "inpath",
            nargs="+",
            help="Path StrainFacts/NetCDF file with one or more parameters.",
        )

    @classmethod
    def run(cls, args):
        inputs = {}
        sample_names = []
        for path in args.inpath:
            mgen = sf.Metagenotype.load(path)
            inputs[path] = mgen
            sample_names.extend(list(mgen.sample.values))
            print(path, mgen.sizes)
        assert len(sample_names) == len(set(sample_names))
        out = sf.data.Metagenotype.concat(inputs, dim="sample", rename=False).mlift(
            "fillna", 0
        )
        out.dump(args.outpath)


class FilterMetagenotype(AppInterface):
    app_name = "filter_mgen"
    description = (
        "Filter metagenotype based on position polymorphism and sample coverage."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--min-minor-allele-freq",
            type=float,
            default=0.05,
            help="Remove sites where less than this fraction of samples have any hits to the minor allele.",
        )
        parser.add_argument(
            "--min-horizontal-cvrg",
            type=float,
            default=0.1,
            help="Remove sample with less than this fraction of sites with non-zero counts.",
        )
        parser.add_argument(
            "inpath",
            help="StrainFacts/NetCDF formatted file to load metagenotype data from.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write StrainFacts/NetCDF formatted file metagenotype data after filtering/subsampling.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        if not (0 <= args.min_minor_allele_freq < 1):
            raise ArgumentConstraintError(
                "Argument --min-minor-allele-freq must be in [0, 1).",
            )
        if not (0 < args.min_horizontal_cvrg < 1):
            raise ArgumentConstraintError(
                "Argument --min-horizontal-cvrg must be between 0 and 1.",
            )
        return args

    @classmethod
    def run(cls, args):
        mgen_all = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {mgen_all.sizes}.")
        mgen_filt = mgen_all.select_variable_positions(
            thresh=args.min_minor_allele_freq
        ).select_samples_with_coverage(args.min_horizontal_cvrg)
        logging.info(f"Output metagenotype shapes: {mgen_filt.sizes}.")
        mgen_filt.dump(args.outpath)


class MetagenotypeDissimilarity(AppInterface):
    app_name = "mgen_diss"
    description = "Calculate pairwise distances over metagenotypes."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "inpath",
            help="StrainFacts/NetCDF formatted file to load metagenotype data from.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write NetCDF formatted dissimilarity matrix.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        pass
        return args

    @classmethod
    def run(cls, args):
        mgen = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {mgen.sizes}.")
        with sf.logging_util.phase_info("Calculating metagenotype dissimilarity."):
            pdist = (
                mgen.pdist()
                .rename_axis(index="sampleA", columns="sampleB")
                .stack()
                .to_xarray()
            )
        pdist.to_dataset(name="dissimilarity").to_netcdf(
            args.outpath,
            engine="netcdf4",
            encoding=dict(dissimilarity=dict(zlib=True, complevel=5)),
        )


class SubsampleMetagenotype(AppInterface):
    app_name = "sample_mgen"
    description = "Select a subsample of metagenotype positions."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--num-positions", type=int, help="Select this number of sites."
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            help="Seed for random number generator; must be set for reproducible subsampling.",
        )
        parser.add_argument(
            "--with-replacement",
            action="store_true",
            default=False,
            help=(
                "Sample with replacement, which allows"
                "sampling more positions than are found in the data."
            ),
        )
        parser.add_argument(
            "--block-number",
            type=int,
            default=0,
            help="Select positions [i * num_positions, (i + 1) * num_positions).",
        )
        parser.add_argument(
            "--no-shuffle",
            dest="shuffle",
            action="store_false",
            help=(
                "Don't randomize positions before selecting "
                "the desired block (therefore, pull a contiguous block of "
                "positions from the original data; by default positions are "
                "NOT contiguous.)"
            ),
        )
        parser.add_argument(
            "--entropy-weighted-alpha",
            type=float,
            default=0.0,
            help=(
                "Sample positions (without replacement) weighted by a "
                "function of their allele entropy across all samples. "
                "Positive values of alpha sample more variable positions, "
                "negative values sample lower entropy positions, and a value "
                "of 0.0 results in unweighted draws."
            ),
        )
        parser.add_argument(
            "inpath",
            help="StrainFacts/NetCDF formatted file to load metagenotype data from.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write StrainFacts/NetCDF formatted file metagenotype data after filtering/subsampling.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        if args.block_number:
            assert args.block_number >= 0
        assert args.num_positions > 0
        if args.with_replacement:
            assert (
                args.block_number == 0
            ), "Block number only makes sense when sampling without replacement."
        # # FIXME: I actually think entropy weighted alpha
        # # is just fine without replacement...as long as
        # # the number of positions is much smaller than
        # # the total number available.
        # if args.entropy_weighted_alpha != 0:
        #     assert (
        #         args.with_replacement
        #     ), "entropy_weighted_alpha not equal to 0 only makes sense when sampling without replacement."
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)

        if not args.with_replacement:
            total_num_positions = metagenotype.sizes["position"]
        else:
            total_num_positions = args.num_positions

        if args.shuffle:
            np.random.seed(args.random_seed)
            logging.info(
                f"Shuffling metagenotype positions using "
                f"random seed {args.random_seed}."
            )
            weight_unnorm = np.exp(
                args.entropy_weighted_alpha * metagenotype.entropy("position").fillna(0)
            )
            weight = weight_unnorm / weight_unnorm.sum()
            position_list = np.random.choice(
                metagenotype.position,
                size=total_num_positions,
                replace=args.with_replacement,
                p=weight,
            )
        else:
            position_list = metagenotype.position

        block_positions = args.num_positions
        block_start = args.block_number * block_positions
        block_stop = min((args.block_number + 1) * block_positions, total_num_positions)
        assert total_num_positions >= block_start
        logging.info(f"Extraction positions [{block_start}, {block_stop}).")

        mgen_ss = metagenotype.sel(position=position_list[block_start:block_stop])
        if args.with_replacement:
            mgen_ss.data["position"] = np.arange(block_start, block_stop)
        mgen_ss.dump(args.outpath)


class Simulate(AppInterface):
    app_name = "simulate"
    description = "Simulate from a metagenotype model."

    @classmethod
    def add_subparser_arguments(cls, parser):
        add_model_structure_cli_argument(parser, default="default_simulation")
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            required=True,
            help="Number of latent strains to simulate.",
        )
        parser.add_argument(
            "--num-samples",
            "-n",
            type=int,
            required=True,
            help="Number of samples to simulate.",
        )
        parser.add_argument(
            "--num-positions",
            "-g",
            type=int,
            required=True,
            help="Number of SNP positions to simulate in (meta)genotype.",
        )
        add_hyperparameters_cli_argument(parser)
        parser.add_argument(
            "--template",
            "-w",
            help="Path to a StrainFacts/NetCDF file with parameter values to be fixed (i.e. conditioning the generative model).",
        )
        parser.add_argument(
            "--fix-from-template",
            default="Which parameters in the template to fix for the simulation.",
        )
        parser.add_argument(
            "--random-seed",
            "--seed",
            "-r",
            type=int,
            help="Seed for random number generator; must be set for reproducible simulations.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write StrainFacts/NetCDF formatted simulated parameters.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = parse_model_structure_string(args.model_structure)
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        args.fix_from_template = args.fix_from_template.split(",")
        return args

    @classmethod
    def run(cls, args):
        if args.template:
            template = sf.data.World.load(args.template)
            data = {k: template[k] for k in args.fix_from_template}
        else:
            data = None

        model, world = sf.workflow.simulate_world(
            structure=args.model_structure,
            sizes=dict(
                strain=args.num_strains,
                sample=args.num_samples,
                position=args.num_positions,
            ),
            hyperparameters=args.hyperparameters,
            data=data,
            seed=args.random_seed,
        )
        world.dump(args.outpath)


class ClusterApproximation(AppInterface):
    app_name = "clust_init"
    description = "Use sample clustering to roughly estimate genotypes."

    @classmethod
    def add_subparser_arguments(cls, parser):
        add_strain_count_cli_arguments(parser)
        parser.add_argument("inpath", help="Metagenotype data input path.")
        parser.add_argument(
            "outpath",
            help="Path to write output StrainFacts/NetCDF file with estimated parameters.",
        )
        # TODO: Figure out how to accept a precalculated dissimilarity matrix.
        parser.add_argument(
            "--frac",
            default=0.5,
            type=float,
            help=(
                "How much strain abundance goes to the cluster strain for "
                "each sample;"
            ),
        )
        parser.add_argument(
            "--pseudo",
            default=1.0,
            type=float,
            help=(
                "Pseudo-count added to metagenotype for consensus genotype "
                "resulting from clusters."
            ),
        )
        parser.add_argument(
            "--thresh",
            default=None,
            type=float,
            help=(
                "Dissimilarity threshold below which to cluster "
                "metagenotypes together; "
                "and may cause a RuntimeError if too many strains are found "
                "in the approximation."
            ),
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args = transform_strain_count_parameter_inputs(args)
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {metagenotype.sizes}.")
        num_strains = calculate_strain_count(metagenotype.sizes["sample"], args)
        world = sf.estimation.clust_approximation(
            metagenotype.to_world(),
            s=num_strains,
            pseudo=args.pseudo,
            frac=args.frac,
            thresh=args.thresh,
            linkage="complete",
        )
        world.dump(args.outpath)


class NMFApproximation(AppInterface):
    app_name = "nmf_init"
    description = "Use NMF to roughly estimate genotypes."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--random-seed",
            "--seed",
            "-r",
            type=int,
            help="Seed for all random number generators; must be set for reproducible model fitting.",
        )
        add_strain_count_cli_arguments(parser)
        parser.add_argument(
            "--alpha-genotype",
            type=float,
            default=0.0,
            help="Regularization parameter on genotype estimate.",
        )
        parser.add_argument(
            "--alpha-community",
            type=float,
            default=0.0,
            help="Regularization parameter on community estimate.",
        )
        parser.add_argument("inpath", help="Metagenotype data input path.")
        parser.add_argument(
            "outpath",
            help="Path to write output StrainFacts/NetCDF file with estimated parameters.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args = transform_strain_count_parameter_inputs(args)
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {metagenotype.sizes}.")
        num_strains = calculate_strain_count(metagenotype.sizes["sample"], args)
        np.random.seed(args.random_seed)
        world = sf.estimation.nmf_approximation(
            metagenotype.to_world(),
            s=num_strains,
            random_state=args.random_seed,
            alpha_W=args.alpha_genotype,
            alpha_H=args.alpha_community,
            l1_ratio=1.0,
            solver="cd",
            tol=1e-3,
            eps=1e-4,
            max_iter=int(1e4),
            init="random",
        )
        world.dump(args.outpath)


class Fit(AppInterface):
    app_name = "fit"
    description = "Fit strain genotypes and community composition to metagenotype data."

    @classmethod
    def add_subparser_arguments(cls, parser):
        add_model_structure_cli_argument(parser)
        add_strain_count_cli_arguments(parser)
        add_hyperparameters_cli_argument(parser)
        parser.add_argument(
            "--tsv",
            action="store_true",
            default=False,
            help="Accept input file in TSV format (rather than StrainFacts/NetCDF).",
        )
        # parser.add_argument("--history-outpath")
        parser.add_argument("inpath", help="Metagenotype data input path.")
        parser.add_argument(
            "outpath",
            help="Path to write output StrainFacts/NetCDF file with estimated parameters.",
        )
        add_optimization_arguments(parser)
        parser.add_argument(
            "--init-from",
            help="A StrainFacts/NetCDF file with genotype and/or community variables to use for initialization.",
        )
        parser.add_argument(
            "--init-vars",
            default=["genotype", "community"],
            choices=["genotype", "community"],
            nargs="+",
            help="Which parameters to use from initialization output; has no effect without one of --nmf-init or --clust-init.",
        )
        parser.add_argument(
            "--anneal-wait",
            type=int,
            default=0,
            help="Number of steps before annealed hyperparameters start stepping.",
        )
        parser.add_argument(
            "--anneal-steps",
            type=int,
            default=0,
            help="Number of steps before annealed hyperparameters are at the their final values; includes `--anneal-wait` steps.",
        )
        parser.add_argument(
            "--anneal-hyperparameters",
            nargs="+",
            action="append",
            default=[],
            help="Values of parameters at the start of optimization, before annealing; arguments are in the form 'NAME=FLOAT'.",
        )
        parser.add_argument(
            "--history-outpath",
            help="Path to record the NLP (loss) value at each step in optimization.",
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = parse_model_structure_string(args.model_structure)
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        args.anneal_hyperparameters = parse_hyperparameter_strings(
            args.anneal_hyperparameters
        )
        if args.anneal_hyperparameters and (args.anneal_steps <= 0):
            raise ArgumentConstraintError(
                "Annealing for 0 steps is like no annealing at all."
            )
        args = transform_strain_count_parameter_inputs(args)
        args = transform_optimization_parameter_inputs(args)
        args.anneal_hyperparameters = {
            k: dict(
                name="log",
                start=args.anneal_hyperparameters[k],
                end=args.hyperparameters[k],
                wait_steps=args.anneal_wait,
            )
            for k in args.anneal_hyperparameters
        }

        return args

    @classmethod
    def run(cls, args):
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sfacts.math",
            lineno=26,
            message="invalid value encountered in float_scalars",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sfacts.model_zoo.components",
            lineno=309,
            message="Using LogTriangle as an approximation for random sampling. This is probably a bad idea.",
        )

        if args.tsv:
            metagenotype = sf.data.Metagenotype.load_from_tsv(args.inpath)
        else:
            metagenotype = sf.data.Metagenotype.load(args.inpath)

        if args.init_from:
            init_from = sf.World.load(args.init_from)
            logging.info(f"Initialization data shapes: {init_from.sizes}.")
        else:
            init_from = None

        num_strains = calculate_strain_count(metagenotype.sizes["sample"], args)

        np.random.seed(args.random_seed)

        logging.debug(
            dict(
                hyperparameters=args.hyperparameters,
                anneal_hyperparameters=args.anneal_hyperparameters,
            )
        )
        est, history = sf.workflow.fit_metagenotype_complex(
            structure=args.model_structure,
            metagenotype=metagenotype,
            nstrain=num_strains,
            init_from=init_from,
            init_vars=args.init_vars,
            hyperparameters=args.hyperparameters,
            anneal_hyperparameters=args.anneal_hyperparameters,
            annealiter=args.anneal_steps,
            device=args.device,
            dtype=args.dtype,
            estimation_kwargs=args.estimation_kwargs,
        )
        est.dump(args.outpath)
        if args.history_outpath:
            with open(args.history_outpath, "w") as f:
                for v in history:
                    print(v, file=f)


class FitGenotypeBlock(AppInterface):
    app_name = "fit_geno"
    description = (
        "Fit strain genotypes based on fixed community composition and metagenotype."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        add_model_structure_cli_argument(parser)
        parser.add_argument(
            "--block-size",
            "-g",
            type=int,
            help="Maximum number of positions fit in each indexed block (independent runs of the program).",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            help="Maximum number of positions fit in each chunk (fit serially in a single run of the program).",
        )
        parser.add_argument(
            "--block-number", "-i", type=int, help="Block index to fit."
        )
        add_hyperparameters_cli_argument(parser)
        # parser.add_argument("--history-outpath")
        parser.add_argument(
            "community",
            help="Previously fit community matrix which the model is conditioned on for refitting.",
        )
        parser.add_argument("metagenotype", help="Metagenotype data input path.")
        parser.add_argument("outpath", help="Path to write estimated genotype matrix.")
        add_optimization_arguments(parser)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = parse_model_structure_string(args.model_structure)
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        if args.block_size is None:
            args.block_size = int(1e20)
            args.block_number = 0
        if args.chunk_size is None:
            args.chunk_size = args.block_size
        args = transform_optimization_parameter_inputs(args)
        return args

    @classmethod
    def run(cls, args):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sfacts.model_zoo.components",
            lineno=309,
            message="Using LogTriangle as an approximation for random sampling. This is probably a bad idea.",
        )
        community = sf.data.World.load(args.community).community
        metagenotype = sf.data.Metagenotype.load(args.metagenotype)

        total_num_positions = metagenotype.sizes["position"]
        block_positions = min(args.block_size, total_num_positions)
        chunk_positions = min(args.chunk_size, total_num_positions)
        block_start = args.block_number * block_positions
        block_stop = min((args.block_number + 1) * block_positions, total_num_positions)
        assert total_num_positions >= block_start
        logging.info(
            (
                f"Selecting genotype block {args.block_number} "
                f"as [{block_start}, {block_stop}) "
                f"from {total_num_positions} positions."
            ),
        )

        metagenotype = metagenotype.mlift(
            "isel", position=slice(block_start, block_stop)
        )

        est, *_ = sf.workflow.iteratively_fit_genotype_conditioned_on_community(
            structure=args.model_structure,
            metagenotype=metagenotype,
            community=community,
            nposition=chunk_positions,
            hyperparameters=args.hyperparameters,
            device=args.device,
            dtype=args.dtype,
            estimation_kwargs=args.estimation_kwargs,
        )
        est.genotype.to_world().dump(args.outpath)


class ConcatGenotypeBlocks(AppInterface):
    app_name = "concat_geno"
    description = "Combine separately fit genotype blocks."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--community",
            help="Path of a community matrix to recombine with concatenated genotypes.",
        )
        parser.add_argument(
            "--metagenotype",
            help="Path of a metagenotype matrix to recombine with concatenated genotypes.",
        )
        parser.add_argument(
            "--outpath",
            required=True,
            help="Path to write the StrainFacts/NetCDF file with recombined components.",
        )
        parser.add_argument(
            "genotypes",
            nargs="+",
            help="One or more genotype matrices to be concatenated and recombined.",
        )

    @classmethod
    def run(cls, args):
        community = sf.data.World.load(args.community).community
        metagenotype = sf.data.Metagenotype.load(args.metagenotype)
        # FIXME: Not clear why metagenotype and genotype have different coordinates (int vs. str).
        metagenotype.data["position"] = metagenotype.data.position.astype(str)
        all_genotypes = {}
        for i, gpath in enumerate(args.genotypes):
            all_genotypes[i] = sf.World.load(gpath).genotype
        all_genotypes = sf.Genotype.concat(all_genotypes, dim="position", rename=False)
        world = sf.World.from_combined(community, metagenotype, all_genotypes)
        assert set(world.position.values) == set(all_genotypes.position.values)
        world.dump(args.outpath)


class CleanInferences(AppInterface):
    app_name = "cleanup_fit"
    description = "Merge similar strains and drop badly fit samples."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--dissimilarity",
            type=float,
            default=0.0,
            help="Distance threshold for clustering.",
        )
        parser.add_argument(
            "--discretized",
            action="store_true",
            help="Discretize genotypes before merging.",
        )
        parser.add_argument(
            "--abundance",
            type=float,
            default=0.0,
            help="Strain minimum max-abundance threshold for strain culling.",
        )
        parser.add_argument(
            "--entropy",
            type=float,
            default=np.inf,
            help="Community entropy threshold for sample culling",
        )
        parser.add_argument(
            "inpath",
            help="Path to StrainFacts/NetCDF file to be cleaned.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write the StrainFacts/NetCDF file with collapsed strains.",
        )

    @classmethod
    def run(cls, args):
        world = sf.World.load(args.inpath)
        logging.info(f"{world.sizes} (input data sizes)")
        world = world.collapse_similar_strains(
            thresh=args.dissimilarity, discretized=args.discretized
        )
        logging.info(f"{world.sizes} (after merging similar strains)")
        world = world.drop_low_abundance_strains(args.abundance)
        logging.info(f"{world.sizes} (after dropping low-abundance strains)")
        world = world.reassign_high_community_entropy_samples(thresh=args.entropy)
        logging.info(
            f"{world.sizes} (after reassigning high community entropy samples)"
        )
        logging.info("Writing output.")
        world.dump(args.outpath)


class DescribeModel(AppInterface):
    app_name = "model_info"
    description = "Summarize a model and its hyperparameters."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "model_structure",
            help="Model name as defined in `sfacts.model_zoo.NAMED_STRUCTURES`.",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            default=3,
            help="Number of strains for model shape description; has no effect with `--shapes`.",
        )
        parser.add_argument(
            "--num-samples",
            "-n",
            type=int,
            default=4,
            help="Number of samples for model shape description; has no effect with `--shapes`.",
        )
        parser.add_argument(
            "--num-positions",
            "-g",
            type=int,
            default=5,
            help="Number of SNP positions for model shape description; has no effect with `--shapes`.",
        )
        parser.add_argument(
            "--shapes",
            action="store_true",
            help="Describe shapes of all model variables.",
        )
        add_hyperparameters_cli_argument(parser)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = parse_model_structure_string(args.model_structure)
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            # module="pyro.poutine.trace_struct",
            lineno=250,
            message="Encountered +inf",
        )
        model = sf.model.ParameterizedModel(
            structure=args.model_structure,
            coords=dict(
                strain=args.num_strains,
                sample=args.num_samples,
                position=args.num_positions,
                allele=["alt", "ref"],
            ),
            hyperparameters=args.hyperparameters,
        )
        print("Defined in:", model.structure.generative.__module__)
        print("Summary:", model.structure.text_summary)
        print("Default Hyperparameters:", model.hyperparameters)
        if args.shapes:
            print(sf.pyro_util.shape_info(model))


class DescribeData(AppInterface):
    app_name = "data_info"
    description = "Summarize shapes from a StrainFacts formatted file."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "inpath",
            nargs="+",
            help="Path StrainFacts/NetCDF file with one or more parameters.",
        )
        parser.add_argument(
            "--header",
            action="store_true",
            help="Print the header line above info for one or more samples.",
        )

    @classmethod
    def run(cls, args):
        keys = ["path"] + list(sf.World.dims)
        if args.header:
            print(*keys, sep="\t")
        for path in args.inpath:
            details = defaultdict(lambda: "")
            details["path"] = path
            _sizes = sf.World.peek_netcdf_sizes(path)
            details.update(_sizes)
            print(
                *[details[k] for k in keys],
                sep="\t",
            )


class Dump(AppInterface):
    app_name = "dump"
    description = "Extract output data from StrainFacts files"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "inpath", help="Path StrainFacts/NetCDF file with one or more parameters."
        )
        parser.add_argument(
            "--nc",
            action="store_true",
            help="Write output to StrainFacts/NetCDF files; otherwise write as TSVs",
        )

        parser.add_argument(
            "--metagenotype",
            help="Path to write metagenotype matrix.",
        )
        parser.add_argument(
            "--community",
            help="Path to write community matrix.",
        )
        parser.add_argument(
            "--genotype",
            help="Path to write genotype matrix.",
        )

    @classmethod
    def run(cls, args):
        world = sf.data.World.load(args.inpath)
        if args.nc:
            _export = lambda var, path: var.dump(path)
        else:
            _export = lambda var, path: var.data.to_series().to_csv(path, sep="\t")

        if args.genotype:
            _export(world.genotype, args.genotype)
        if args.metagenotype:
            _export(world.metagenotype, args.metagenotype)
        if args.community:
            _export(world.community, args.community)


class EvaluateFitAgainstSimulation(AppInterface):
    app_name = "evaluate_fit"
    description = "Calculate goodness-of-fit scores between estimates and ground-truth."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "reference",
            help="Path to StrainFacts/NetCDF file with reference data for comparison.",
        )
        parser.add_argument(
            "--simulation",
            action="store_true",
            help='Reference includes "ground-truth" community and genotype, i.e. from a simulation',
        )
        parser.add_argument(
            "fit",
            nargs="+",
            help="Path(s) to one or more StrainFacts/NetCDF files with estimated parameters.",
        )
        parser.add_argument(
            "--outpath",
            help="Write TSV of evaluation scores to file; otherwise write to STDOUT.",
        )
        parser.add_argument(
            "--num-format", help="Python format string for writing all numbers."
        )
        parser.add_argument(
            "--transpose",
            action="store_true",
            help="Transpose rows and columns of output.",
        )

    @classmethod
    def run(cls, args):
        ref = sf.World.load(args.reference)
        results = {}
        for fit_path in args.fit:
            fit = sf.World.load(fit_path)
            # FIXME: dtype of the coordinates changes from int to 'object'
            # (string) at some point during processing.
            # NOTE: This fix is just a hack and is probably fairly brittle.
            fit = sf.World(fit.data.assign_coords(position=fit.position.astype(int)))
            metrics = sf.workflow.evaluate_fit_against_metagenotype(ref, fit)
            if args.simulation:
                metrics.update(sf.workflow.evaluate_fit_against_simulation(ref, fit))
            results[fit_path] = metrics

        results = pd.DataFrame(results).rename_axis(index="score")

        if args.outpath:
            outpath_or_handle = args.outpath
        else:
            outpath_or_handle = sys.stdout

        if args.transpose:
            results = results.T

        results.to_csv(
            outpath_or_handle,
            sep="\t",
            index=True,
            header=True,
            float_format=args.num_format,
        )


SUBCOMMANDS = [
    # Debugging
    DescribeModel,
    DescribeData,
    # Input/Output
    Load,
    Dump,
    # Data Processing
    StackMetagenotypes,
    FilterMetagenotype,
    MetagenotypeDissimilarity,
    SubsampleMetagenotype,
    ConcatGenotypeBlocks,
    CleanInferences,
    # Simulation:
    Simulate,
    # Fitting:
    ClusterApproximation,
    NMFApproximation,
    Fit,
    FitGenotypeBlock,
    # Evaluation:
    EvaluateFitAgainstSimulation,
]


def main():
    logging.basicConfig(format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("--version", action="version", version=sf.__version__)

    app_subparsers = parser.add_subparsers(metavar="COMMAND")
    for subcommand in SUBCOMMANDS:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    if not hasattr(args, "_subcommand"):
        print(parser.format_help())
        # args._subcommand = Help
    else:
        args._subcommand(args)
