#!/usr/bin/env python3

import sys
import argparse
import warnings
import xarray as xr
import torch
import sfacts as sf
import numpy as np
import importlib
from copy import deepcopy
import itertools


def parse_hyperparameter_strings(list_of_lists_of_pairs):
    list_of_pairs = itertools.chain.from_iterable(list_of_lists_of_pairs)
    hyperparameters = {}
    for pair in list_of_pairs:
        key, value = pair.split("=", 2)
        hyperparameters[key] = float(value)
    return hyperparameters


def add_optimization_arguments(parser):
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=sf.pyro_util.PRECISION_MAP.keys(),
        help="Float precision.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-iter", default=int(1e5), type=int)
    parser.add_argument("--random-seed", "--seed", "-r", type=int)
    parser.add_argument("--lag1", default=20, type=int)
    parser.add_argument("--lag2", default=100, type=int)
    parser.add_argument("--nojit", dest="jit", action="store_false", default=True)
    parser.add_argument(
        "--optimizer", default="Adamax", choices=sf.estimation.OPTIMIZERS.keys()
    )
    parser.add_argument("--optimizer-learning-rate", type=float)
    parser.add_argument(
        "--min-optimizer-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate threshold to stop reduction 'schedule'.",
    )


def transform_optimization_parameter_inputs(args):
    args = deepcopy(args)
    optimizer_kwargs = {}
    if args.optimizer_learning_rate is not None:
        optimizer_kwargs["lr"] = args.optimizer_learning_rate

    args.estimation_kwargs = dict(
        seed=args.random_seed,
        jit=args.jit,
        ignore_jit_warnings=True,
        maxiter=args.max_iter,
        lagA=args.lag1,
        lagB=args.lag2,
        optimizer_name=args.optimizer,
        optimizer_kwargs=optimizer_kwargs,
        minimum_lr=args.min_optimizer_learning_rate,
    )
    return args


class AppInterface:
    app_name = "TODO"
    description = "TODO"

    @classmethod
    def add_subparser_arguments(cls, subparser):
        raise NotImplementedError(
            "Subclasses of AppInterface must implement a `add_subparser_arguments` method."
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        raise NotImplementedError(
            "Subclasses of AppInterface must implement a `finalize_input_arguments` method."
        )

    @classmethod
    def run(cls, args):
        raise NotImplementedError(
            "Subclasses of AppInterface must implement a `run` method."
        )

    def __init__(self, args):
        """Run the application."""
        args = self.transform_app_parameter_inputs(deepcopy(args))
        self.run(args)

    @classmethod
    def _add_app_subparser(cls, app_subparsers):
        subparser = app_subparsers.add_parser(cls.app_name, help=cls.description)
        subparser.set_defaults(_subcommand=cls)
        cls.add_subparser_arguments(subparser)


class NoOp(AppInterface):
    app_name = "do_nothing"
    description = "dummy subcommand that does nothing"

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("--dummy", type=int, default=1)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        pass

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        print(args)


class FilterMetagenotypes(AppInterface):
    app_name = "filter_mgen"
    description = (
        "Filter metagenotypes based on position polymorphism and sample coverage."
    )

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument("--min-minor-allele-freq", type=float, default=0.05)
        parser.add_argument("--min-horizontal-cvrg", type=float, default=0.1)
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
            default="full_metagenotype",
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

        parser.add_argument("--outpath", "-o", required=True)

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


class FitSimple(AppInterface):
    app_name = "simple_fit"
    description = "Simply estimate parameters of a metagenotype model given data."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--model-structure",
            "-m",
            default="full_metagenotype",
            help="See sfacts.model_zoo.__init__.NAMED_STRUCTURES",
            choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
        )
        parser.add_argument("--num-strains", "-s", type=int, required=True)
        parser.add_argument("--num-positions", "-g", type=int)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        parser.add_argument("--anneal-wait", type=int, default=0)
        parser.add_argument("--anneal-steps", type=int, default=0)
        parser.add_argument("--anneal-hyperparameters", nargs="+", default=[])
        add_optimization_arguments(parser)
        parser.add_argument(
            "--tsv",
            action="store_true",
            default=False,
            help="Input file is in TSV format (rather than NetCDF).",
        )
        parser.add_argument("--verbose", "-v", action="store_true", default=False)
        parser.add_argument("--history-outpath")
        parser.add_argument("inpath")
        parser.add_argument("outpath")

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        args.anneal_hyperparameters = {
            k: dict(name="log", start=1.0, end=args.hyperparameters[k], wait_steps=args.anneal_wait)
            for k in args.anneal_hyperparameters
        }
        args = transform_optimization_parameter_inputs(args)
        return args

    @classmethod
    def run(cls, args):
        if args.tsv:
            metagenotypes = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        else:
            metagenotypes = sf.data.Metagenotypes.load(args.inpath)

        np.random.seed(args.random_seed)
        if args.num_positions:
            metagenotypes = metagenotypes.random_sample(args.num_positions, "position")
        est, history = sf.workflow.fit_metagenotypes_simple(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            nstrain=args.num_strains,
            hyperparameters=args.hyperparameters,
            anneal_hyperparameters=args.anneal_hyperparameters,
            annealiter=args.anneal_steps,
            device=args.device,
            dtype=sf.pyro_util.PRECISION_MAP[args.precision],
            quiet=(not args.verbose),
            estimation_kwargs=args.estimation_kwargs,
        )
        est.dump(args.outpath)

        if args.history_outpath:
            with open(args.history_outpath, "w") as f:
                for elbo in history:
                    print(elbo, file=f)


class FitComplex(AppInterface):
    app_name = "complex_fit"
    description = "Fit data using some tricky tactics."

    @classmethod
    def add_subparser_arguments(cls, parser):
        parser.add_argument(
            "--model-structure",
            "-m",
            default="full_metagenotype",
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
                "Fix initial strain number. "
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
        parser.add_argument("--verbose", "-v", action="store_true", default=False)
        parser.add_argument("--history-outpath")
        parser.add_argument("inpath")
        parser.add_argument("outpath")
        add_optimization_arguments(parser)
        parser.add_argument(
            "--collapse",
            default=0.0,
            type=float,
            help="Dissimilarity threshold to collapse highly similar strains.",
        )
        parser.add_argument(
            "--cull",
            default=0.0,
            type=float,
            help="Minimum single-sample abundance to keep a strain.",
        )
        parser.add_argument("--anneal-wait", type=int, default=0)
        parser.add_argument("--anneal-steps", type=int, default=0)
        parser.add_argument("--anneal-hyperparameters", nargs="+", default=[])
        parser.add_argument(
            "--refinement-hyperparameters", nargs="+", action="append", default=[]
        )

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        if args.num_strains and args.strains_per_sample:
            raise Exception(
                "Only one of --num-strains or --strains-per-sample may be set."
            )
        elif (args.num_strains is None) and (args.strains_per_sample is None):
            raise Exception(
                "One of either --num-strains or --strains-per-sample must be set."
            )
        args = transform_optimization_parameter_inputs(args)
        del args.estimation_kwargs['seed']  # Here consumed by workflow, not estimation.
        args.anneal_hyperparameters = {
            k: dict(name="log", start=1.0, end=args.hyperparameters[k], wait_steps=args.anneal_wait)
            for k in args.anneal_hyperparameters
        }
        args.refinement_hyperparameters = parse_hyperparameter_strings(args.refinement_hyperparameters)
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
                np.ceil(metagenotypes.sizes["sample"] * args.strains_per_sample)
            )
        else:
            num_strains = args.num_strains

        (
            est,
            est_list,
            history_list,
        ) = sf.workflow.fit_subsampled_metagenotypes_then_collapse_and_iteratively_refit_genotypes(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            nstrain=num_strains,
            nposition=args.num_positions,
            diss_thresh=args.collapse,
            frac_thresh=args.cull,
            hyperparameters=args.hyperparameters,
            anneal_hyperparameters=args.anneal_hyperparameters,
            annealiter=args.anneal_steps,
            stage2_hyperparameters=args.refinement_hyperparameters,
            device=args.device,
            dtype=sf.pyro_util.PRECISION_MAP[args.precision],
            quiet=(not args.verbose),
            seed=args.random_seed,
            estimation_kwargs=args.estimation_kwargs,
        )
        est.dump(args.outpath)
        history = history_list[0]

        if args.history_outpath:
            with open(args.history_outpath, "w") as f:
                for elbo in history:
                    print(elbo, file=f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    app_subparsers = parser.add_subparsers()
    for subcommand in [NoOp, FilterMetagenotypes, Simulate, FitSimple, FitComplex]:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    args._subcommand(args)
