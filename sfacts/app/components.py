import sfacts as sf
import itertools
from copy import deepcopy
import argparse


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
    parser.add_argument("--max-iter", default=int(1e6), type=int)
    parser.add_argument("--random-seed", "--seed", "-r", type=int)
    parser.add_argument("--lag1", default=50, type=int)
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
    parser.add_argument("--optimizer-clip-norm", type=float)


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
    def run(cls, args):
        raise NotImplementedError(
            "Subclasses of AppInterface must implement a `run` method."
        )

    def __init__(self, args):
        """Run the application."""
        args = self.transform_app_parameter_inputs(deepcopy(args))
        if args.debug:
            sf.logging_util.info(args)
        self.run(args)

    @classmethod
    def _add_app_subparser(cls, app_subparsers):
        subparser = app_subparsers.add_parser(
            cls.app_name,
            help=cls.description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subparser.set_defaults(_subcommand=cls)
        subparser.add_argument("--verbose", "-v", action="store_true", default=False)
        subparser.add_argument("--debug", action="store_true", default=False)
        cls.add_subparser_arguments(subparser)
