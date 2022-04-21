import argparse
import sys
import warnings
import sfacts as sf
import numpy as np
import pandas as pd
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
            "--dummy", type=int, default=1, help="test option; does nothing"
        )
        add_hyperparameters_cli_argument(parser)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        print(args)


class Load(AppInterface):
    app_name = "load"
    description = "Build StrainFacts files from input data."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("--gtpro-metagenotype")
        parser.add_argument("--metagenotype")
        parser.add_argument("--community")
        parser.add_argument("--genotype")
        parser.add_argument("outpath")

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
                sf.data.Metagenotypes.load_from_merged_gtpro(args.gtpro_metagenotype)
            )
        if args.metagenotype:
            values.append(sf.data.Metagenotypes.load_from_tsv(args.metagenotype))

        if args.community:
            values.append(sf.data.Communities.load_from_tsv(args.community))
        if args.genotype:
            values.append(sf.data.Genotypes.load_from_tsv(args.genotype))
        world = sf.data.World.from_combined(*values)
        world.dump(args.outpath)


class FilterMetagenotypes(AppInterface):
    app_name = "filter_mgen"
    description = (
        "Filter metagenotypes based on position polymorphism and sample coverage."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--min-minor-allele-freq", type=float, default=0.05, help=" "
        )
        parser.add_argument("--min-horizontal-cvrg", type=float, default=0.1, help=" ")
        parser.add_argument("--num-positions", type=int)
        parser.add_argument("--random-seed", type=int)
        parser.add_argument("inpath")
        parser.add_argument("outpath")

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
        mgen_all = sf.data.Metagenotypes.load(args.inpath)
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
        add_model_structure_cli_argument(parser)
        parser.add_argument("--num-strains", "-s", type=int, required=True)
        parser.add_argument("--num-samples", "-n", type=int, required=True)
        parser.add_argument("--num-positions", "-g", type=int, required=True)
        add_hyperparameters_cli_argument(parser)
        parser.add_argument("--template", "-w")
        parser.add_argument("--fix-from-template", default="")
        parser.add_argument("--random-seed", "--seed", "-r", type=int)

        parser.add_argument("outpath")

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
            help="Dynamically set strain number as a fraction of sample number.",
        )
        parser.add_argument(
            "--num-strains",
            "-s",
            type=int,
            help=(
                "Fix strain number. "
                "(Only one of --num-strains or --strains-per-sample may be set)"
            ),
        )
        parser.add_argument("--num-positions", "-g", type=int)
        add_hyperparameters_cli_argument(parser)
        parser.add_argument(
            "--tsv",
            action="store_true",
            default=False,
            help="Input file is in TSV format (rather than NetCDF).",
        )
        # parser.add_argument("--history-outpath")
        parser.add_argument("inpath")
        parser.add_argument("outpath")
        add_optimization_arguments(parser)
        parser.add_argument(
            "--no-nmf-init",
            dest="nmf_init",
            action="store_false",
            default=True,
            help="don't use NMF to select starting parameters (do use NMF by default)",
        )
        parser.add_argument("--anneal-wait", type=int, default=0)
        parser.add_argument("--anneal-steps", type=int, default=0)
        parser.add_argument(
            "--anneal-hyperparameters", nargs="+", action="append", default=[]
        )
        parser.add_argument("--history-outpath")

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
        if args.num_strains and args.strains_per_sample:
            raise argparse.ArgumentError(
                "strains_per_sample",
                "Only one of --num-strains or --strains-per-sample may be set.",
            )
        if (args.num_strains is None) and (args.strains_per_sample is None):
            raise argparse.ArgumentError(
                "num_strains",
                "One of either --num-strains or --strains-per-sample must be set.",
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
            metagenotypes = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        else:
            metagenotypes = sf.data.Metagenotypes.load(args.inpath)

        if args.strains_per_sample:
            num_strains = int(
                max(np.ceil(metagenotypes.sizes["sample"] * args.strains_per_sample), 2)
            )
        else:
            num_strains = args.num_strains

        np.random.seed(args.random_seed)
        num_positions_ss = min(args.num_positions, metagenotypes.sizes["position"])
        metagenotypes_ss = metagenotypes.random_sample(position=num_positions_ss)

        if args.debug:
            sf.logging_util.info("\n\n")
            sf.logging_util.info(
                dict(
                    hyperparameters=args.hyperparameters,
                    anneal_hyperparameters=args.anneal_hyperparameters,
                )
            )
        est0, est_list, history_list = sf.workflow.fit_metagenotypes_complex(
            structure=args.model_structure,
            metagenotypes=metagenotypes_ss,
            nstrain=num_strains,
            nmf_init=args.nmf_init,
            nmf_seed=args.random_seed,
            hyperparameters=args.hyperparameters,
            anneal_hyperparameters=args.anneal_hyperparameters,
            annealiter=args.anneal_steps,
            device=args.device,
            dtype=args.dtype,
            quiet=(not args.verbose),
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
        "Fit strain genotypes based on fixed community compositions and metagenotypes."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        add_model_structure_cli_argument(parser)
        parser.add_argument("--block-size", "-g", type=int)
        parser.add_argument("--chunk-size", type=int)
        parser.add_argument("--block-number", "-i", type=int)
        add_hyperparameters_cli_argument(parser)
        # parser.add_argument("--history-outpath")
        parser.add_argument("community")
        parser.add_argument("metagenotype")
        parser.add_argument("outpath")
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
        communities = sf.data.World.load(args.community).communities
        metagenotypes = sf.data.Metagenotypes.load(args.metagenotype)

        total_num_positions = metagenotypes.sizes["position"]
        block_positions = min(args.block_size, total_num_positions)
        chunk_positions = min(args.chunk_size, total_num_positions)
        block_start = args.block_number * block_positions
        block_stop = min((args.block_number + 1) * block_positions, total_num_positions)
        assert total_num_positions >= block_start
        sf.logging_util.info(
            (
                f"Selecting genotype block {args.block_number} "
                f"as [{block_start}, {block_stop}) "
                f"from {total_num_positions} positions."
            ),
            quiet=(not args.verbose),
        )

        metagenotypes = metagenotypes.mlift(
            "isel", position=slice(block_start, block_stop)
        )

        est, *_ = sf.workflow.iteratively_fit_genotypes_conditioned_on_communities(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            communities=communities,
            nposition=chunk_positions,
            hyperparameters=args.hyperparameters,
            device=args.device,
            dtype=args.dtype,
            quiet=(not args.verbose),
            estimation_kwargs=args.estimation_kwargs,
        )
        est.genotypes.to_world().dump(args.outpath)


class ConcatGenotypeBlocks(AppInterface):
    app_name = "concat_geno"
    description = "Combine separately fit genotype blocks."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("--community")
        parser.add_argument("--metagenotype")
        parser.add_argument("--outpath", required=True)
        parser.add_argument("genotypes", nargs="+")

    @classmethod
    def run(cls, args):
        communities = sf.data.World.load(args.community).communities
        metagenotypes = sf.data.Metagenotypes.load(args.metagenotype)
        # FIXME: Not clear why metagenotypes and genotypes have different coordinates (int vs. str).
        metagenotypes.data["position"] = metagenotypes.data.position.astype(str)
        all_genotypes = {}
        for i, gpath in enumerate(args.genotypes):
            all_genotypes[i] = sf.World.load(gpath).genotypes
        all_genotypes = sf.Genotypes.concat(all_genotypes, dim="position", rename=False)
        world = sf.World.from_combined(communities, metagenotypes, all_genotypes)
        assert set(world.position.values) == set(all_genotypes.position.values)
        world.dump(args.outpath)


class DebugModel(AppInterface):
    app_name = "describe"
    description = (
        "List the hyperparameters of the model and their default (or CLI set) values."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "model_structure",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument("--num-strains", "-s", type=int, default=3)
        parser.add_argument("--num-samples", "-n", type=int, default=4)
        parser.add_argument("--num-positions", "-g", type=int, default=5)
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
        sf.logging_util.info(model.hyperparameters)
        if args.shapes:
            sf.pyro_util.shape_info(model)


class Dump(AppInterface):
    app_name = "dump"
    description = "Extract output data from StrainFacts files"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("inpath")
        parser.add_argument(
            "--nc",
            action="store_true",
            help="write output to NetCDF files; otherwise TSVs",
        )
        parser.add_argument("--genotype")
        parser.add_argument("--community")
        parser.add_argument("--metagenotype")

    @classmethod
    def run(cls, args):
        world = sf.data.World.load(args.inpath)
        if args.nc:
            _export = lambda var, path: var.dump(path)
        else:
            _export = lambda var, path: var.data.to_series().to_csv(path, sep="\t")

        if args.genotype:
            _export(world.genotypes, args.genotype)
        if args.metagenotype:
            _export(world.metagenotypes, args.metagenotype)
        if args.community:
            _export(world.communities, args.community)


# class DumpMetagenotype(AppInterface):
#     app_name = "dump_mgen"
#     description = "Export metagenotype to TSV"
#
#     @classmethod
#     def add_subparser_arguments(cls, parser):
#         parser.add_argument("inpath")
#         parser.add_argument("outpath")
#
#     @classmethod
#     def run(cls, args):
#         mgen = sf.data.Metagenotypes.load(args.inpath)
#         _export = lambda var, path: var.data.to_series().to_csv(path, sep="\t")
#         _export(mgen, args.outpath)


class EvaluateFitAgainstSimulation(AppInterface):
    app_name = "evaluate_fit"
    description = "Calculate goodness-of-fit scores between estimates and ground-truth."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("simulation")
        parser.add_argument("fit", nargs="+")
        parser.add_argument(
            "--outpath", help="Write evaluation scores to file; otherwise STDOUT"
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

        results = pd.DataFrame(results).rename_axis(index='score')

        if args.outpath:
            outpath_or_handle = args.outpath
        else:
            outpath_or_handle = sys.stdout
        results.to_csv(
            outpath_or_handle, sep="\t", index=True, header=True
        )


SUBCOMMANDS = [
    # Debugging
    NoOp,
    DescribeModel,
    # Input/Output
    Load,
    Dump,
    # Data Processing
    FilterMetagenotypes,
    ConcatGenotypeBlocks,
    # Simulation:
    Simulate,
    # Fitting:
    Fit,
    FitGenotypeBlock,
    # Evaluation:
    EvaluateFitAgainstSimulation,
]


def main():
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
