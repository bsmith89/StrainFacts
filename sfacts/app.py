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
    parser.add_argument("--optimizer-learning-rate", default=1e-1, type=float)
    parser.add_argument("--optimizer-clip", default=1e2, type=float)


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
        parser.add_argument("--random-seed", "--seed", "-r", type=int)
        parser.add_argument("--outpath", "-o", required=True)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        model, world = sf.workflow.simulate_world(
            structure=args.model_structure,
            sizes=dict(
                strain=args.num_strains,
                sample=args.num_samples,
                position=args.num_positions,
            ),
            hyperparameters=args.hyperparameters,
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
        parser.add_argument("--random-seed", "--seed", "-r", type=int)
        add_optimization_arguments(parser)
        parser.add_argument("--inpath", "-i", required=True)
        parser.add_argument("--outpath", "-o", required=True)
        parser.add_argument("--verbose", "-v", action="store_true", default=False)

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        metagenotypes = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        if args.num_positions:
            metagenotypes = metagenotypes.random_sample(args.num_positions, "position")
        est, history = sf.workflow.fit_metagenotypes_simple(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            nstrain=args.num_strains,
            hyperparameters=args.hyperparameters,
            device=args.device,
            dtype=sf.pyro_util.PRECISION_MAP[args.precision],
            quiet=(not args.verbose),
            estimation_kwargs=dict(
                seed=args.random_seed,
                ignore_jit_warnings=True,
                maxiter=args.max_iter,
                optimizer_kwargs=dict(
                    optim_args={"lr": args.optimizer_learning_rate},
                    clip_args={"clip_norm": args.optimizer_clip},
                ),
            ),
        )
        est.dump(args.outpath)


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
            "--strains_per_sample",
            "-s",
            type=float,
            required=True,
            help="Dynamically set strain number as a fraction of sample number.",
        )
        parser.add_argument("--num-positions", "-g", type=int)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        parser.add_argument("--random-seed", "--seed", "-r", type=int)
        parser.add_argument("--inpath", "-i", required=True)
        parser.add_argument("--outpath", "-o", required=True)
        parser.add_argument("--verbose", "-v", action="store_true", default=False)
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

    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        metagenotypes = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        if args.num_positions is not None:
            metagenotypes = metagenotypes.random_sample(position=args.num_positions)
        num_strains = int(
            np.ceil(metagenotypes.sizes["sample"] * args.strains_per_sample)
        )
        est = sf.workflow.fit_subsampled_metagenotype_collapse_strains_then_iteratively_refit_full_genotypes(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            nstrain=num_strains,
            diss_thresh=args.collapse,
            frac_thresh=args.cull,
            hyperparameters=args.hyperparameters,
            stage2_hyperparameters=dict(gamma_hyper=1.0),
            device=args.device,
            dtype=sf.pyro_util.PRECISION_MAP[args.precision],
            quiet=(not args.verbose),
            estimation_kwargs=dict(
                seed=args.random_seed,
                ignore_jit_warnings=True,
                maxiter=args.max_iter,
                optimizer_kwargs=dict(
                    optim_args={"lr": args.optimizer_learning_rate},
                    clip_args={"clip_norm": args.optimizer_clip},
                ),
            ),
        )
        est.dump(args.outpath)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    app_subparsers = parser.add_subparsers()
    for subcommand in [NoOp, Simulate, FitSimple, FitComplex]:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    args._subcommand(args)
