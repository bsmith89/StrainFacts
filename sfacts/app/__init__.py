import argparse
import warnings
import sfacts as sf
import numpy as np
from sfacts.app.components import (
    parse_hyperparameter_strings,
    add_optimization_arguments,
    transform_optimization_parameter_inputs,
    AppInterface,
)


class NoOp(AppInterface):
    app_name = "do_nothing"
    description = "dummy subcommand that does nothing"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("--dummy", type=int, default=1)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        print(args)


class LoadMetagenotype(AppInterface):
    app_name = "load_mgen"
    description = "Read in metagentype TSV and convert it into a StrainFacts NetCDF metagenotype file."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("inpath")
        parser.add_argument("--gtpro", action="store_true")
        parser.add_argument("outpath")

    @classmethod
    def run(cls, args):
        if args.gtpro:
            mgen = sf.data.Metagenotypes.load_from_merged_gtpro(args.inpath)
        else:
            mgen = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        mgen.dump(args.outpath)


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
        assert 0 < args.min_minor_allele_freq < 1
        assert 0 < args.min_horizontal_cvrg < 1
        if args.num_positions is None:
            args.num_positions = int(1e20)
        return args

    @classmethod
    def run(cls, args):
        mgen_all = sf.data.Metagenotypes.load(args.inpath)
        mgen_filt = mgen_all.select_variable_positions(
            thresh=0.05
        ).select_samples_with_coverage(0.05)

        nposition = min(mgen_filt.sizes["position"], args.num_positions)
        np.random.seed(args.random_seed)
        mgen_filt_ss = mgen_filt.random_sample(position=nposition)
        mgen_filt_ss.dump(args.outpath)


class Simulate(AppInterface):
    app_name = "simulate"
    description = "Simulate from a metagenotype model."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--model-structure",
            "-m",
            default="default_simulation",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument("--num-strains", "-s", type=int, required=True)
        parser.add_argument("--num-samples", "-n", type=int, required=True)
        parser.add_argument("--num-positions", "-g", type=int, required=True)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        parser.add_argument("--template", "-w")
        parser.add_argument("--fix-from-template", default="")
        parser.add_argument("--random-seed", "--seed", "-r", type=int)

        parser.add_argument("outpath")

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
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
        parser.add_argument(
            "--model-structure",
            "-m",
            default="default",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
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
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
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
            "--nmf-init",
            dest="nmf_init",
            action="store_true",
            default=True,
            help="Use NMF to select starting parameters.",
        )
        parser.add_argument(
            "--no-nmf-init",
            dest="nmf_init",
            action="store_false",
            help="Don't use NMF to select starting parameters.",
        )
        parser.add_argument("--anneal-wait", type=int, default=0)
        parser.add_argument("--anneal-steps", type=int, default=0)
        parser.add_argument(
            "--anneal-hyperparameters", nargs="+", action="append", default=[]
        )
        parser.add_argument("--history-outpath")

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        args.anneal_hyperparameters = parse_hyperparameter_strings(
            args.anneal_hyperparameters
        )
        if args.anneal_hyperparameters:
            assert (
                args.anneal_steps > 0
            ), "Annealing for 0 steps is like no annealing at all."
        if args.num_strains and args.strains_per_sample:
            raise Exception(
                "Only one of --num-strains or --strains-per-sample may be set."
            )
        elif (args.num_strains is None) and (args.strains_per_sample is None):
            raise Exception(
                "One of either --num-strains or --strains-per-sample must be set."
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
        assert num_strains > 1

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
    app_name = "fit_genotype"
    description = (
        "Fit strain genotypes based on fixed community compositions and metagenotypes."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--model-structure",
            "-m",
            default="default",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument("--num-positions", "-g", type=int)
        parser.add_argument("--num-positionsB", type=int)
        parser.add_argument("--block-number", "-i", type=int)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        # parser.add_argument("--history-outpath")
        parser.add_argument("community")
        parser.add_argument("metagenotype")
        parser.add_argument("outpath")
        add_optimization_arguments(parser)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        if args.num_positions is None:
            args.num_positions = int(1e20)
        if args.num_positionsB is None:
            args.num_positionsB = args.num_positions
        args = transform_optimization_parameter_inputs(args)
        return args

    @classmethod
    def run(cls, args):
        communities = sf.data.World.load(args.community).communities
        metagenotypes = sf.data.Metagenotypes.load(args.metagenotype)

        total_num_positions = metagenotypes.sizes["position"]
        num_positions = min(args.num_positions, metagenotypes.sizes["position"])
        num_positionsB = min(args.num_positionsB, metagenotypes.sizes["position"])
        block_start = args.block_number * num_positions
        block_stop = min((args.block_number + 1) * num_positions, total_num_positions)
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
            nposition=num_positionsB,
            hyperparameters=args.hyperparameters,
            device=args.device,
            dtype=args.dtype,
            quiet=(not args.verbose),
            estimation_kwargs=args.estimation_kwargs,
        )
        est.genotypes.to_world().dump(args.outpath)


class ConcatGenotypeBlocks(AppInterface):
    app_name = "concatenate_genotype_chunks"
    description = "Combine step of a split-apply-combine workflow."

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


class DebugModelHyperparameters(AppInterface):
    app_name = "debug_hypers"
    description = (
        "List the hyperparameters of the model and their set (or default) values."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "model_structure",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        model = sf.model.ParameterizedModel(
            structure=args.model_structure,
            coords=dict(
                strain=2,
                sample=2,
                position=2,
                allele=["alt", "ref"],
            ),
            hyperparameters=args.hyperparameters,
        )
        print(model.hyperparameters)


class SplitWorld(AppInterface):
    app_name = "split"
    description = "Export individual StrainFacts parameters"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("inpath")
        parser.add_argument("--genotype")
        parser.add_argument("--community")
        parser.add_argument("--metagenotype")

    @classmethod
    def run(cls, args):
        world = sf.data.World.load(args.inpath)
        if args.genotype:
            world.genotypes.dump(args.genotype)
        if args.metagenotype:
            world.metagenotypes.dump(args.metagenotype)
        if args.community:
            world.communities.dump(args.community)


class Dump(AppInterface):
    app_name = "dump"
    description = "Export StrainFacts parameters to individual files"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("inpath")
        parser.add_argument("--tsv", action="store_true")
        parser.add_argument("--genotype")
        parser.add_argument("--community")
        parser.add_argument("--metagenotype")

    @classmethod
    def run(cls, args):
        world = sf.data.World.load(args.inpath)
        if args.tsv:
            _export = lambda var, path: var.data.to_series().to_csv(path, sep="\t")
        else:
            _export = lambda var, path: var.dump(path)

        if args.genotype:
            _export(world.genotypes, args.genotype)
        if args.metagenotype:
            _export(world.metagenotypes, args.metagenotype)
        if args.community:
            _export(world.communities, args.community)


class DumpMetagenotype(AppInterface):
    app_name = "dump_mgen"
    description = "Export metagenotype to TSV"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("inpath")
        parser.add_argument("outpath")

    @classmethod
    def run(cls, args):
        mgen = sf.data.Metagenotypes.load(args.inpath)
        _export = lambda var, path: var.data.to_series().to_csv(path, sep="\t")
        _export(mgen, args.outpath)


class EvaluateFitAgainstSimulation(AppInterface):
    app_name = "evaluate_fit"
    description = "TODO"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("simulation")
        parser.add_argument("fit")
        parser.add_argument("outpath")

    @classmethod
    def run(cls, args):
        fit = sf.World.load(args.fit)
        # FIXME: dtype of the coordinates changes from int to 'object' (string) at some point during processing.
        # NOTE: This fix is just a hack and is probably fairly brittle.
        fit = sf.World(fit.data.assign_coords(position=fit.position.astype(int)))
        sim = sf.World.load(args.simulation)

        errors = sf.workflow.evaluate_fit_against_simulation(sim, fit)
        errors.to_csv(args.outpath, sep="\t", index=True, header=False)


SUBCOMMANDS = [
    # Debugging
    NoOp,
    DebugModelHyperparameters,
    # Input/Output
    LoadMetagenotype,
    DumpMetagenotype,
    Dump,
    # DataProcessing
    FilterMetagenotypes,
    ConcatGenotypeBlocks,
    # Simulation
    Simulate,
    # Fitting
    Fit,
    FitGenotypeBlock,
    # Evaluation
    EvaluateFitAgainstSimulation,
]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    app_subparsers = parser.add_subparsers()
    for subcommand in SUBCOMMANDS:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    if not hasattr(args, "_subcommand"):
        print(parser.format_help())
        # args._subcommand = Help
    else:
        args._subcommand(args)
