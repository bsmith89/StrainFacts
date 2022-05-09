import sfacts as sf
import itertools
from copy import deepcopy
import argparse
import logging


def add_hyperparameters_cli_argument(parser):
    parser.add_argument(
        "--hyperparameters",
        "-p",
        nargs="+",
        action="append",
        default=[],
        help="List of model hyperparameters to override defaults; arguments are in the form 'NAME=FLOAT'.",
    )


def parse_hyperparameter_strings(list_of_lists_of_pairs):
    list_of_pairs = itertools.chain.from_iterable(list_of_lists_of_pairs)
    hyperparameters = {}
    for pair in list_of_pairs:
        key, value = pair.split("=", 2)
        hyperparameters[key] = float(value)
    return hyperparameters


def add_model_structure_cli_argument(parser, default="default"):
    parser.add_argument(
        "--model-structure",
        "-m",
        default=default,
        help="Model name as defined in `sfacts.model_zoo.NAMED_STRUCTURES`.",
        choices=sf.model_zoo.NAMED_STRUCTURES.keys(),
    )


def parse_model_structure_string(model_structure):
    return sf.model_zoo.NAMED_STRUCTURES[model_structure]


def add_optimization_arguments(parser):
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=sf.pyro_util.PRECISION_MAP.keys(),
        help="Float precision.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Load model onto a specific PyTorch device.",
    )
    parser.add_argument(
        "--max-iter",
        default=int(1e6),
        type=int,
        help="Maximum number of optimization steps.",
    )
    parser.add_argument(
        "--random-seed",
        "--seed",
        "-r",
        type=int,
        help="Seed for all random number generators; must be set for reproducible model fitting.",
    )
    parser.add_argument(
        "--lag1",
        default=50,
        type=int,
        help="Setting for `pyro.optim.ReduceLROnPlateau` 'cooldown' argument.",
    )
    parser.add_argument(
        "--lag2",
        default=100,
        type=int,
        help="Setting for `pyro.optim.ReduceLROnPlateau` 'patience' argument.",
    )
    parser.add_argument(
        "--no-jit",
        dest="jit",
        action="store_false",
        default=True,
        help="Don't use the PyTorch JIT; much slower steps, but no warm-up; may be useful for debugging.",
    )
    parser.add_argument(
        "--optimizer",
        default="Adamax",
        choices=sf.estimation.OPTIMIZERS.keys(),
        help="Which Pyro optimizer to use.",
    )
    parser.add_argument(
        "--optimizer-learning-rate",
        type=float,
        help="Set the optimizer learning rate; otherwise use the default set in `sfacts.estimation.OPTIMIZERS`.",
    )
    parser.add_argument(
        "--min-optimizer-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate threshold in reduction 'schedule' to terminate optimization.",
    )
    parser.add_argument(
        "--optimizer-clip-norm",
        type=float,
        help="Set the clip_norm for Pyro optimizer; otherwise default is None",
    )


def transform_optimization_parameter_inputs(args):
    args = deepcopy(args)
    optimizer_kwargs = {}
    args.dtype = sf.pyro_util.PRECISION_MAP[args.precision]
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
        optimizer_clip_kwargs=dict(clip_norm=args.optimizer_clip_norm),
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
        return args

    @classmethod
    def _setup_logging(cls, args):
        if args.debug:
            logging_level = logging.DEBUG
        elif args.verbose:
            logging_level = logging.INFO
        else:
            logging_level = logging.WARNING
        logging.getLogger().setLevel(logging_level)
        logging.debug(f"Set logging level to {logging_level}")

    @classmethod
    def run(cls, args):
        raise NotImplementedError(
            "Subclasses of AppInterface must implement a `run` method."
        )

    def __init__(self, args):
        """Run the application."""
        args = self.transform_app_parameter_inputs(deepcopy(args))
        self._setup_logging(args)
        logging.debug(args)
        self.run(args)

    @classmethod
    def _add_app_subparser(cls, app_subparsers):
        subparser = app_subparsers.add_parser(
            cls.app_name,
            help=cls.description,
            description=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparser.set_defaults(_subcommand=cls)
        subparser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Print info messages to stderr.",
        )
        subparser.add_argument(
            "--debug",
            action="store_true",
            help="Print debug messages to stderr.",
        )
        cls.add_subparser_arguments(subparser)
