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
            raise argparse.ArgumentError(
                "gtpro_metagenotype",
                "Only one of --num-strains or --strains-per-sample may be set.",
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
            "--num-positions", type=int, help="Randomly subsample this number of sites."
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
            raise argparse.ArgumentError(
                "min_minor_allele_freq",
                "Argument min_minor_allele_freq must be between 0 and 1.",
            )
        if not (0 < args.min_horizontal_cvrg < 1):
            raise argparse.ArgumentError(
                "min_horizontal_cvrg",
                "Argument min_horizontal_cvrg must be between 0 and 1.",
            )
        if args.num_positions is None:
            args.num_positions = int(1e20)
        return args

    @classmethod
    def run(cls, args):
        mgen_all = sf.data.Metagenotype.load(args.inpath)
        mgen_filt = mgen_all.select_variable_positions(
            thresh=args.min_minor_allele_freq
        ).select_samples_with_coverage(args.min_horizontal_cvrg)

        nposition = min(mgen_filt.sizes["position"], args.num_positions)
        np.random.seed(args.random_seed)
        mgen_filt_ss = mgen_filt.random_sample(position=nposition)
        mgen_filt_ss.dump(args.outpath)


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
        parser.add_argument(
            "--num-positions",
            "-g",
            type=int,
            help="Number of randomly subsampled SNP positions from metagenotype to fit.",
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
            "--init-vars",
            default=["genotype", "community"],
            nargs="+",
            help="Which parameters to use from initialization output; has no effect without one of --nmf-init or --clust-init.",
        )
        parser.add_argument(
            "--nmf-init",
            dest="nmf_init",
            action="store_true",
            default=False,
            help="Use NMF to select starting parameters.",
        )
        parser.add_argument(
            "--clust-init",
            dest="clust_init",
            action="store_true",
            default=False,
            help="Use agglomorative clustering to select starting parameters.",
        )
        parser.add_argument(
            "--clust-init-frac",
            default=0.5,
            type=float,
            help=(
                "How much strain abundance goes to the cluster strain for "
                "each sample; has no effect without --clust-init"
            ),
        )
        parser.add_argument(
            "--clust-init-pseudo",
            default=1.0,
            type=float,
            help=(
                "Pseudo-count added to metagenotype for consensus genotype "
                "resulting from clusters. has no effect without --clust-init"
            ),
        )
        parser.add_argument(
            "--clust-init-thresh",
            default=None,
            type=float,
            help=(
                "Dissimilarity threshold below which to cluster "
                "metagenotypes together; has no effect without --clust-init; "
                "and may cause a RuntimeError if too many strains are found "
                "in the approximation."
            ),
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
            raise argparse.ArgumentError(
                "anneal_steps", "Annealing for 0 steps is like no annealing at all."
            )
        strain_setting_indicator = [
            int(bool(getattr(args, k)))
            for k in ["num_strains", "strains_per_sample", "strain_sample_exponent"]
        ]
        if sum(strain_setting_indicator) != 1:
            raise argparse.ArgumentError(
                "num_strains",
                "One and only one of --num-strains, --strain-sample-exponent or --strains-per-sample may be set.",
            )
        if args.num_strains and (args.num_strains < 2):
            raise argparse.ArgumentError(
                "num_strains", "num_strains must be 2 or more."
            )
        if args.num_positions is None:
            args.num_positions = int(1e20)
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

        # Initialization
        if args.nmf_init and args.clust_init:
            raise argparse.ArgumentError(
                "nmf_init", "Only one of --nmf-init and --clust-init may be used."
            )
        if args.nmf_init:
            args.init_func = sf.estimation.nmf_approximation
            args.init_kwargs = dict(
                random_state=args.random_seed,
                alpha=0.0,
                l1_ratio=1.0,
                solver="cd",
                tol=1e-3,
                eps=1e-4,
                max_iter=int(1e4),
                init="random",
            )
        elif args.clust_init:
            args.init_func = sf.estimation.clust_approximation
            args.init_kwargs = dict(
                pseudo=args.clust_init_pseudo,
                frac=args.clust_init_frac,
                thresh=args.clust_init_thresh,
                linkage="average",
            )
        else:
            args.init_func = None
            args.init_kwargs = dict()
        return args

    @classmethod
    def run(cls, args):
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="sfacts.math",
            lineno=43,
            message="Progress bar not implemented for genotype_pdist.",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sfacts.math",
            lineno=26,
            message="invalid value encountered in float_scalars",
        )

        if args.tsv:
            metagenotype = sf.data.Metagenotype.load_from_tsv(args.inpath)
        else:
            metagenotype = sf.data.Metagenotype.load(args.inpath)

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
        num_positions_ss = min(args.num_positions, metagenotype.sizes["position"])
        metagenotype_ss = metagenotype.random_sample(position=num_positions_ss)

        if args.debug:
            logging.info("\n\n")
            logging.info(
                dict(
                    hyperparameters=args.hyperparameters,
                    anneal_hyperparameters=args.anneal_hyperparameters,
                )
            )
        est0, est_list, history_list = sf.workflow.fit_metagenotype_complex(
            structure=args.model_structure,
            metagenotype=metagenotype_ss,
            nstrain=num_strains,
            init_vars=args.init_vars,
            init_func=args.init_func,
            init_kwargs=args.init_kwargs,
            hyperparameters=args.hyperparameters,
            anneal_hyperparameters=args.anneal_hyperparameters,
            annealiter=args.anneal_steps,
            device=args.device,
            dtype=args.dtype,
            estimation_kwargs=args.estimation_kwargs,
        )
        est0.dump(args.outpath)
        if args.history_outpath:
            with open(args.history_outpath, "w") as f:
                for v in history_list[0]:
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
        print("Hyperparameters:", model.hyperparameters)
        if args.shapes:
            print(sf.pyro_util.shape_info(model))


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
            "simulation",
            help="Path to StrainFacts/NetCDF file with ground-truth parameters.",
        )
        parser.add_argument(
            "fit",
            nargs="+",
            help="Path(s) to one or more StrainFacts/NetCDF files with estimated parameters.",
        )
        parser.add_argument(
            "--outpath",
            help="Write TSV of evaluation scores to file; otherwise to STDOUT",
        )

    @classmethod
    def run(cls, args):
        sim = sf.World.load(args.simulation)
        results = {}
        for fit_path in args.fit:
            fit = sf.World.load(fit_path)
            # FIXME: dtype of the coordinates changes from int to 'object'
            # (string) at some point during processing.
            # NOTE: This fix is just a hack and is probably fairly brittle.
            fit = sf.World(fit.data.assign_coords(position=fit.position.astype(int)))
            results[fit_path] = sf.workflow.evaluate_fit_against_simulation(sim, fit)

        results = pd.DataFrame(results).rename_axis(index="score")

        if args.outpath:
            outpath_or_handle = args.outpath
        else:
            outpath_or_handle = sys.stdout
        results.to_csv(outpath_or_handle, sep="\t", index=True, header=True)


SUBCOMMANDS = [
    # Debugging
    NoOp,
    DescribeModel,
    # Input/Output
    Load,
    Dump,
    # Data Processing
    FilterMetagenotype,
    ConcatGenotypeBlocks,
    CollapseStrains,
    # Simulation:
    Simulate,
    # Fitting:
    Fit,
    FitGenotypeBlock,
    # Evaluation:
    EvaluateFitAgainstSimulation,
]


def main():
    logging.basicConfig()
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
