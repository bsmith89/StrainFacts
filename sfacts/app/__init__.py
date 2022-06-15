import argparse
import sys
import warnings
import sfacts as sf
import numpy as np
import pandas as pd
import logging
from sfacts.app.components import (
    add_hyperparameters_cli_argument,
    parse_hyperparameter_strings,
    add_model_structure_cli_argument,
    parse_model_structure_string,
    add_optimization_arguments,
    transform_optimization_parameter_inputs,
    AppInterface,
)


class ArgumentConstraintError(Exception):
    pass


class ArgumentMutualExclusionError(Exception):
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
            "--random-seed",
            type=int,
            help="Seed for random number generator; must be set for reproducible subsampling.",
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
        if not (0 < args.min_minor_allele_freq < 1):
            raise ArgumentConstraintError(
                "Argument --min-minor-allele-freq must be between 0 and 1.",
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
                "positions from the original data)."
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
        assert args.num_positions > 0
        if args.block_number:
            assert args.block_number >= 0
        if args.num_positions is None:
            args.num_positions = int(1e20)
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)

        total_num_positions = metagenotype.sizes["position"]
        num_positions = min(args.num_positions, total_num_positions)

        if args.shuffle:
            np.random.seed(args.random_seed)
            logging.info(
                f"Shuffling metagenotype positions using random seed {args.random_seed}."
            )
            position_list = np.random.choice(
                metagenotype.position, size=total_num_positions, replace=False
            )
        else:
            position_list = metagenotype.position

        block_positions = args.num_positions
        block_start = args.block_number * block_positions
        block_stop = min((args.block_number + 1) * block_positions, total_num_positions)
        assert total_num_positions >= block_start
        logging.info(f"Extraction positions [{block_start}, {block_stop}).")

        mgen_ss = metagenotype.sel(position=position_list[block_start:block_stop])
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
        parser.add_argument(
            "--random-seed",
            "--seed",
            "-r",
            type=int,
            help="Seed for all random number generators; must be set for reproducible model fitting.",
        )
        parser.add_argument(
            "--strains-per-sample",
            type=float,
            help=(
                "Set number of latent strains to this fixed ratio with the number of samples "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--strain-sample-exponent",
            type=float,
            help=(
                "Set number of latent strains to the number of samples raised to this exponent "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            help=(
                "Set number of latent strains to fit "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument("inpath", help="Metagenotype data input path.")
        parser.add_argument(
            "outpath",
            help="Path to write output StrainFacts/NetCDF file with estimated parameters.",
        )
        parser.add_argument(
            "--frac",
            default=0.5,
            type=float,
            help=(
                "How much strain abundance goes to the cluster strain for "
                "each sample; has no effect without --clust-init"
            ),
        )
        parser.add_argument(
            "--pseudo",
            default=1.0,
            type=float,
            help=(
                "Pseudo-count added to metagenotype for consensus genotype "
                "resulting from clusters. has no effect without --clust-init"
            ),
        )
        parser.add_argument(
            "--thresh",
            default=None,
            type=float,
            help=(
                "Dissimilarity threshold below which to cluster "
                "metagenotypes together; has no effect without --clust-init; "
                "and may cause a RuntimeError if too many strains are found "
                "in the approximation."
            ),
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        strain_setting_indicator = [
            int(bool(getattr(args, k)))
            for k in ["num_strains", "strains_per_sample", "strain_sample_exponent"]
        ]
        if sum(strain_setting_indicator) != 1:
            raise ArgumentMutualExclusionError(
                "One and only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set.",
            )
        if args.num_strains and (args.num_strains < 2):
            raise ArgumentConstraintError("--num-strains must be 2 or more.")
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {metagenotype.sizes}.")

        if args.strains_per_sample:
            num_strains = int(
                max(np.ceil(metagenotype.sizes["sample"] * args.strains_per_sample), 2)
            )
        elif args.strain_sample_exponent:
            num_strains = int(
                max(
                    np.ceil(
                        metagenotype.sizes["sample"] ** args.strain_sample_exponent
                    ),
                    2,
                )
            )
        else:
            num_strains = args.num_strains

        np.random.seed(args.random_seed)

        world = sf.estimation.clust_approximation(
            metagenotype.to_world(),
            s=num_strains,
            pseudo=args.pseudo,
            frac=args.frac,
            thresh=args.thresh,
            linkage="average",
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
        parser.add_argument(
            "--strains-per-sample",
            type=float,
            help=(
                "Set number of latent strains to this fixed ratio with the number of samples "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--strain-sample-exponent",
            type=float,
            help=(
                "Set number of latent strains to the number of samples raised to this exponent "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            help=(
                "Set number of latent strains to fit "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
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
        strain_setting_indicator = [
            int(bool(getattr(args, k)))
            for k in ["num_strains", "strains_per_sample", "strain_sample_exponent"]
        ]
        if sum(strain_setting_indicator) != 1:
            raise ArgumentMutualExclusionError(
                "One and only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set.",
            )
        if args.num_strains and (args.num_strains < 2):
            raise ArgumentConstraintError("Argument --num-strains must be 2 or more.")
        return args

    @classmethod
    def run(cls, args):
        metagenotype = sf.data.Metagenotype.load(args.inpath)
        logging.info(f"Input metagenotype shapes: {metagenotype.sizes}.")

        if args.strains_per_sample:
            num_strains = int(
                max(np.ceil(metagenotype.sizes["sample"] * args.strains_per_sample), 2)
            )
        elif args.strain_sample_exponent:
            num_strains = int(
                max(
                    np.ceil(
                        metagenotype.sizes["sample"] ** args.strain_sample_exponent
                    ),
                    2,
                )
            )
        else:
            num_strains = args.num_strains

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
        parser.add_argument(
            "--strains-per-sample",
            type=float,
            help=(
                "Set number of latent strains to this fixed ratio with the number of samples "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--strain-sample-exponent",
            type=float,
            help=(
                "Set number of latent strains to the number of samples raised to this exponent "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            help=(
                "Set number of latent strains to fit "
                "(only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set)."
            ),
        )
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
        strain_setting_indicator = [
            int(bool(getattr(args, k)))
            for k in ["num_strains", "strains_per_sample", "strain_sample_exponent"]
        ]
        if sum(strain_setting_indicator) != 1:
            raise ArgumentMutualExclusionError(
                "One and only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set.",
            )
        if args.num_strains and (args.num_strains < 2):
            raise ArgumentConstraintError("Argument --num-strains must be 2 or more.")
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

        if args.strains_per_sample:
            num_strains = int(
                max(np.ceil(metagenotype.sizes["sample"] * args.strains_per_sample), 2)
            )
        elif args.strain_sample_exponent:
            num_strains = int(
                max(
                    np.ceil(
                        metagenotype.sizes["sample"] ** args.strain_sample_exponent
                    ),
                    2,
                )
            )
        else:
            num_strains = args.num_strains

        np.random.seed(args.random_seed)

        logging.debug("\n\n")
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


class CollapseStrains(AppInterface):
    app_name = "collapse_strains"
    description = "Merge similar strains."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--discretized",
            action="store_true",
            help="Discretize genotypes before clustering.",
        )
        parser.add_argument(
            "thresh",
            type=float,
            help="Distance threshold for clustering.",
        )
        parser.add_argument(
            "inpath",
            help="Path to StrainFacts/NetCDF file to be collapsed.",
        )
        parser.add_argument(
            "outpath",
            help="Path to write the StrainFacts/NetCDF file with collapsed strains.",
        )

    @classmethod
    def run(cls, args):
        world = sf.World.load(args.inpath)
        world_collapsed = world.collapse_strains(
            thresh=args.thresh, discretized=args.discretized
        )
        world_collapsed.dump(args.outpath)


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

    @classmethod
    def run(cls, args):
        for path in args.inpath:
            world = sf.World.load(path)
            print(path, world.sizes)
        # for dim in world.dims:
        #     print('{}: {}'.format(dim, world.sizes[dim]))


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
    SubsampleMetagenotype,
    ConcatGenotypeBlocks,
    CollapseStrains,
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

    app_subparsers = parser.add_subparsers(metavar="COMMAND")
    for subcommand in SUBCOMMANDS:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    if not hasattr(args, "_subcommand"):
        print(parser.format_help())
        # args._subcommand = Help
    else:
        args._subcommand(args)
