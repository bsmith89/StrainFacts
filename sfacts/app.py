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
        parser.add_argument("--num_strains", "-s", type=int, required=True)
        parser.add_argument("--num_samples", "-n", type=int, required=True)
        parser.add_argument("--num_positions", "-g", type=int, required=True)
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


class Fit(AppInterface):
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
        parser.add_argument("--num_strains", "-s", type=int, required=True)
        parser.add_argument(
            "--hyperparameters", "-p", nargs="+", action="append", default=[]
        )
        parser.add_argument("--random-seed", "--seed", "-r", type=int)
        parser.add_argument(
            "--precision",
            type=int,
            default=32,
            choices=sf.pyro_util.PRECISION_MAP.keys(),
            help="Float precision.",
        )
        parser.add_argument("--inpath", "-i", required=True)
        parser.add_argument("--outpath", "-o", required=True)
        parser.add_argument("--verbose", "-v", action='store_true', default=False)
        parser.add_argument("--device", default='cpu')


    @classmethod
    def transform_app_parameter_inputs(cls, args):
        args.model_structure = sf.model_zoo.NAMED_STRUCTURES[args.model_structure]
        args.hyperparameters = parse_hyperparameter_strings(args.hyperparameters)
        return args

    @classmethod
    def run(cls, args):
        metagenotypes = sf.data.Metagenotypes.load_from_tsv(args.inpath)
        est, history = sf.workflow.fit_metagenotypes_simple(
            structure=args.model_structure,
            metagenotypes=metagenotypes,
            nstrain=args.num_strains,
            hyperparameters=args.hyperparameters,
            device=args.device,
            dtype=sf.pyro_util.PRECISION_MAP[args.precision],
            quiet=(not args.verbose),
            estimation_kwargs=dict(seed=args.random_seed, ignore_jit_warnings=True),
        )
        est.dump(args.outpath)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    app_subparsers = parser.add_subparsers()
    for subcommand in [NoOp, Simulate, Fit]:
        subcommand._add_app_subparser(app_subparsers)

    args = parser.parse_args()
    args._subcommand(args)
