# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys

import dpctl
import dpnp as np
import numpy

from kmeans_python import kmeans_python
from generate_random_data import gen_rand_data
#from generate_random_data import SEED

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

NUMBER_OF_CENTROIDS = 10

###############################################


def printCentroid(arrayC, arrayCsum, arrayCnumpoint, NUMBER_OF_CENTROIDS):
    for i in range(NUMBER_OF_CENTROIDS):
        print(
            "[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
                arrayC[i, 0],
                arrayC[i, 1],
                arrayCsum[i, 0],
                arrayCsum[i, 1],
                arrayCnumpoint[i],
            )
        )

    print("--------------------------------------------------")


def gen_data_np(nopt):
    X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_rand_data(
        nopt, dtype=np.float32
    )
    return (X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint)

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


def gen_data_dpnp(nopt):
    X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_data_np(nopt)

    #convert to dpnp
    return (to_dpnp(X), to_dpnp(arrayPclusters), to_dpnp(arrayC), to_dpnp(arrayCsum), to_dpnp(arrayCnumpoint))


##############################################


def run(name, alg, sizes=6, step=2, nopt=2**17):
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
    ndims = 2
    niters = 30

    dpctl.SyclDevice("gpu")
    
    f = open("perf_output.csv", "w")
    f2 = open("runtimes.csv", "w", 1)

    if args.test:
        (
            X,
            arrayPclusters_p,
            arrayC_p,
            arrayCsum_p,
            arrayCnumpoint_p,
        ) = gen_data_np(nopt)
        kmeans_python(
            X,
            arrayPclusters_p,
            arrayC_p,
            arrayCsum_p,
            arrayCnumpoint_p,
            nopt,
            NUMBER_OF_CENTROIDS,
        )

        (
            X_n,
            arrayPclusters_n,
            arrayC_n,
            arrayCsum_n,
            arrayCnumpoint_n,
        ) = gen_data_dpnp(nopt)

        # pass numpy generated data to kernel
        alg(
            X_n,
            arrayPclusters_n,
            arrayC_n,
            arrayCsum_n,
            arrayCnumpoint_n,
            niters,
            nopt,
            ndims,
            NUMBER_OF_CENTROIDS,
        )

        if (
            np.allclose(arrayC_n, arrayC_p)
            and np.allclose(arrayCsum_n, arrayCsum_p)
            and np.allclose(arrayCnumpoint_n, arrayCnumpoint_p)
        ):
            print(
                "Test succeeded\n",
                "arrayC_Python:",
                arrayC_p,
                "\n arrayC_numba:",
                arrayC_n,
                "arrayCsum_python:",
                arrayCsum_p,
                "\n arracyCsum_numba:",
                arrayCsum_n,
                "arrayCnumpoint_python:",
                arrayCnumpoint_p,
                "\n arrayCnumpoint_numba:",
                arrayCnumpoint_n,
            )
        else:
            print(
                "Test failed\n",
                "arrayC_Python:",
                arrayC_p,
                "\n arrayC_numba:",
                arrayC_n,
                "arrayCsum_python:",
                arrayCsum_p,
                "\n arracyCsum_numba:",
                arrayCsum_n,
                "arrayCnumpoint_python:",
                arrayCnumpoint_p,
                "\n arrayCnumpoint_numba:",
                arrayCnumpoint_n,
            )
        return

    for i in xrange(sizes):
        X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_data_dpnp(
            nopt
        )

        iterations = xrange(repeat)
        sys.stdout.flush()

        alg(
            X,
            arrayPclusters,
            arrayC,
            arrayCsum,
            arrayCnumpoint,
            niters,
            nopt,
            ndims,
            NUMBER_OF_CENTROIDS,
        )  # warmup
        t0 = now()
        for _ in iterations:
            alg(
                X,
                arrayPclusters,
                arrayC,
                arrayCsum,
                arrayCnumpoint,
                niters,
                nopt,
                ndims,
                NUMBER_OF_CENTROIDS,
            )

        mops, time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * 2 * repeat, time
            ),
            flush=True,
        )
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
    f2.close()