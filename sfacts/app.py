#!/usr/bin/env python3

import sys
import argparse
import warnings
import xarray as xr
import torch
from sfacts.logging_util import info
import numpy as np


def parse_args(argv):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input
    p.add_argument(
        "pileup",
        nargs="+",
        help="""
Single, fully processed, pileup table in NetCDF format
with the following dimensions:
    * library_id
    * position
    * allele
                        """,
    )

    # Shape of the model
    p.add_argument("--nstrains", metavar="INT", type=int, default=1000)
    p.add_argument(
        "--npos",
        metavar="INT",
        default=2000,
        type=int,
        help=("Number of positions to sample for model fitting."),
    )

    # Data filtering
    p.add_argument(
        "--incid-thresh",
        metavar="FLOAT",
        type=float,
        default=0.02,
        help=(
            "Minimum fraction of samples that must have the minor allele "
            "for the position to be considered 'informative'."
        ),
    )
    p.add_argument(
        "--cvrg-thresh",
        metavar="FLOAT",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of 'informative' positions with counts "
            "necessary for sample to be included."
        ),
    )

    # Regularization
    p.add_argument(
        "--gamma-hyper",
        metavar="FLOAT",
        default=1e-2,
        type=float,
        help=("Ambiguity regularization parameter."),
    )
    p.add_argument(
        "--pi-hyper",
        metavar="FLOAT",
        default=1e-1,
        type=float,
        help=("Heterogeneity regularization parameter (will be scaled by 1 / s)."),
    )
    p.add_argument(
        "--rho-hyper",
        metavar="FLOAT",
        default=1e0,
        type=float,
        help=("Diversity regularization parameter."),
    )
    p.add_argument("--epsilon-hyper", metavar="FLOAT", default=0.01, type=float)
    p.add_argument(
        "--epsilon",
        metavar="FLOAT",
        default=None,
        type=float,
        help=("Fixed error rate for all samples."),
    )
    p.add_argument("--alpha-hyper", metavar="FLOAT", default=100.0, type=float)
    p.add_argument(
        "--alpha",
        metavar="FLOAT",
        default=None,
        type=float,
        help=("Fixed concentration for all samples."),
    )
    p.add_argument(
        "--collapse",
        metavar="FLOAT",
        default=0.0,
        type=float,
        help=("Merge strains with a cosine distance of less than this value."),
    )

    # Fitting
    p.add_argument("--random-seed", default=0, type=int, help=("FIXME"))
    p.add_argument("--max-iter", default=10000, type=int, help=("FIXME"))
    p.add_argument("--lag", default=50, type=int, help=("FIXME"))
    p.add_argument("--stop-at", default=5.0, type=float, help=("FIXME"))
    p.add_argument("--learning-rate", default=1e-0, type=float, help=("FIXME"))
    p.add_argument("--clip-norm", default=100.0, type=float, help=("FIXME"))

    # Hardware
    p.add_argument("--device", default="cpu", help=("PyTorch device name."))

    # Output
    p.add_argument(
        "--outpath",
        metavar="PATH",
        help=("Path for genotype fraction output."),
    )

    args = p.parse_args(argv)

    return args


if __name__ == "__main__":
    warnings.filterwarnings(
        "error",
        message="Encountered NaN: loss",
        category=UserWarning,
        # module="trace_elbo",  # FIXME: What is the correct regex for module?
        lineno=217,
    )
    warnings.filterwarnings(
        "ignore",
        message="CUDA initialization: Found no NVIDIA",
        category=UserWarning,
        lineno=130,
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.tensor results are registered as constants",
        category=torch.jit.TracerWarning,
        # module="trace_elbo",  # FIXME: What is the correct regex for module?
        lineno=95,
    )

    # args = parse_args(sys.argv[1:])
    # info(args)
    #
    # info(f"Setting random seed: {args.random_seed}")
    # np.random.seed(args.random_seed)
    #
    # info("Loading input data.")
    # data = _load_input_data(args.pileup)
    # info(f"Full data shape: {data.sizes}.")
    #
    # info("Filtering positions.")
    # informative_positions = select_informative_positions(data, args.incid_thresh)
    # npos_available = len(informative_positions)
    # info(
    #     f"Found {npos_available} informative positions with minor "
    #     f"allele incidence of >{args.incid_thresh}"
    # )
    # npos = min(args.npos, npos_available)
    # info(f"Randomly sampling {npos} positions.")
    # position_ss = np.random.choice(
    #     informative_positions,
    #     size=npos,
    #     replace=False,
    # )
    #
    # info("Filtering libraries.")
    # suff_cvrg_samples = idxwhere(
    #     (
    #         (data.sel(position=informative_positions).sum(["allele"]) > 0).mean(
    #             "position"
    #         )
    #         > args.cvrg_thresh
    #     ).to_series()
    # )
    # nlibs = len(suff_cvrg_samples)
    # info(
    #     f"Found {nlibs} libraries with >{args.cvrg_thresh:0.1%} "
    #     f"of informative positions covered."
    # )
    #
    # info("Constructing input data.")
    # data_fit = data.sel(library_id=suff_cvrg_samples, position=position_ss)
    # m_ss = data_fit.sum("allele")
    # n, g_ss = m_ss.shape
    # y_obs_ss = data_fit.sel(allele="alt")
    #
    # info("Optimizing model parameters.")
    # mapest1, history1 = find_map(
    #     model_fit,
    #     init=as_torch_all(
    #         gamma=init_genotype,
    #         pi=init_frac,
    #         dtype=torch.float32,
    #         device=args.device,
    #     ),
    #     lag=args.lag,
    #     stop_at=args.stop_at,
    #     learning_rate=args.learning_rate,
    #     max_iter=args.max_iter,
    #     clip_norm=args.clip_norm,
    # )
    # if args.device.startswith("cuda"):
    #     info(
    #         "CUDA available mem: {}".format(
    #             torch.cuda.get_device_properties(0).total_memory
    #         ),
    #     )
    #     info("CUDA reserved mem: {}".format(torch.cuda.memory_reserved(0)))
    #     info("CUDA allocated mem: {}".format(torch.cuda.memory_allocated(0)))
    #     info(
    #         "CUDA free mem: {}".format(
    #             torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    #         )
    #     )
    #     torch.cuda.empty_cache()
    #
    # info("Finished fitting model.")
    # result = xr.Dataset(
    #     {
    #         "gamma": (["strain", "position"], mapest3["gamma"]),
    #         "rho": (["strain"], mapest3["rho"]),
    #         "alpha_hyper": ([], mapest3["alpha_hyper"]),
    #         "pi": (["library_id", "strain"], mapest3["pi"]),
    #         "epsilon": (["library_id"], mapest3["epsilon"]),
    #         "rho_hyper": ([], mapest3["rho_hyper"]),
    #         "epsilon_hyper": ([], mapest3["epsilon_hyper"]),
    #         "pi_hyper": ([], mapest3["pi_hyper"]),
    #         "alpha": (["library_id"], mapest3["alpha"]),
    #         "p_noerr": (["library_id", "position"], mapest3["p_noerr"]),
    #         "p": (["library_id", "position"], mapest3["p"]),
    #         "y": (["library_id", "position"], y_obs_ss),
    #         "m": (["library_id", "position"], m_ss),
    #         "elbo_trace": (["iteration"], history1),
    #     },
    #     coords=dict(
    #         strain=np.arange(s_collapse),
    #         position=data_fit.position,
    #         library_id=data_fit.library_id,
    #     ),
    # )
    #
    # if args.outpath:
    #     info("Saving results.")
    #     result.to_netcdf(
    #         args.outpath,
    #         encoding=dict(
    #             gamma=dict(dtype="float32", zlib=True, complevel=6),
    #             pi=dict(dtype="float32", zlib=True, complevel=6),
    #             p_noerr=dict(dtype="float32", zlib=True, complevel=6),
    #             p=dict(dtype="float32", zlib=True, complevel=6),
    #             y=dict(dtype="uint16", zlib=True, complevel=6),
    #             m=dict(dtype="uint16", zlib=True, complevel=6),
    #             elbo_trace=dict(dtype="float32", zlib=True, complevel=6),
    #         ),
    #     )
