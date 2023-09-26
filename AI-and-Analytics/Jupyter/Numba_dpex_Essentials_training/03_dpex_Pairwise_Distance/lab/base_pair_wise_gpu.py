# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy
import dpctl
import dpnp as np
from generate_data_random import gen_rand_data
import dpctl, dpctl.tensor as dpt
from pairwise_distance_python import (
    pairwise_distance_python,
)

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################
# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################
def gen_data(nopt, dims):
    X, Y = gen_rand_data(nopt, dims)
    return (X, Y, numpy.empty((nopt, nopt)))

def to_dpnp(ref_array):
    if ref_array.flags["C_CONTIGUOUS"]:
        order = "C"
    elif ref_array.flags["F_CONTIGUOUS"]:
        order = "F"
    else:
        order = "K"
    return np.asarray(
        ref_array,
        dtype=ref_array.dtype,
        order=order,
        like=None,
        device="gpu",
        usm_type=None,
        sycl_queue=None,
    )

def to_numpy(ref_array):
    return np.asnumpy(ref_array)

def gen_data_dpnp(nopt, dims):
    X, Y = gen_rand_data(nopt, dims)
    return (to_dpnp(X), to_dpnp(Y), np.empty((nopt, nopt)))

##############################################


def run(name, alg, sizes=7, step=2, nopt=2**10):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", required=False, default=sizes, help="Number of steps"
    )
    parser.add_argument(
        "--step", required=False, default=step, help="Factor for each step"
    )
    parser.add_argument(
        "--size", required=False, default=nopt, help="Initial data size"
    )
    parser.add_argument(
        "--repeat",
        required=False,
        default=1,
        help="Iterations inside measured region",
    )
    parser.add_argument(
        "--text", required=False, default="", help="Print with each result"
    )
    parser.add_argument("-d", type=int, default=3, help="Dimensions")
    parser.add_argument(
        "--test",
        required=False,
        action="store_true",
        help="Check for correctness by comparing output with naieve Python version",
    )

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    dpctl.SyclDevice("gpu")

    if args.test:
        X, Y, p_D = gen_data(nopt, dims)
        pairwise_distance_python(X, Y, p_D)

        n_X, n_Y, n_D = gen_data_dpnp(nopt, dims)
        alg(n_X, n_Y, n_D)

        if np.allclose(n_D, p_D):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        X, Y, D = gen_data_dpnp(nopt, dims)

        iterations = xrange(repeat)

        alg(X, Y, D)  # warmup
        t0 = now()
        for _ in iterations:
            alg(X, Y, D)

        mops, time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * repeat, time
            ),
            flush=True,
        )
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
    f2.close()