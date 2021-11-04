# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
import sys, json, os
import dpctl, dpctl.tensor as dpt
from pairwise_distance_python import (
    pairwise_distance_python,
)
from generate_data_random import gen_rand_data

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
def get_device_selector(is_gpu=True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if (
        os.environ.get("SYCL_DEVICE_FILTER") is None
        or os.environ.get("SYCL_DEVICE_FILTER") == "opencl"
    ):
        return "opencl:" + device_selector

    if os.environ.get("SYCL_DEVICE_FILTER") == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get("SYCL_DEVICE_FILTER")


def gen_data(nopt, dims):
    X, Y = gen_rand_data(nopt, dims)
    return (X, Y, np.empty((nopt, nopt)))


def gen_data_usm(nopt, dims):
    X, Y, D = gen_data(nopt, dims)

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        X_usm = dpt.usm_ndarray(
            X.shape,
            dtype=X.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        Y_usm = dpt.usm_ndarray(
            Y.shape,
            dtype=Y.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        D_usm = dpt.usm_ndarray(
            D.shape,
            dtype=D.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    X_usm.usm_data.copy_from_host(X.reshape((-1)).view("u1"))
    Y_usm.usm_data.copy_from_host(Y.reshape((-1)).view("u1"))

    return (X_usm, Y_usm, D_usm)


##############################################

def write_dictionary(dictionary, fn):
    import pickle
    import os
    here = './'
    # Store data (serialize)
    with open(os.path.join(here,fn), 'wb') as handle:
        #pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dictionary, handle)
    return

def read_dictionary(fn):
    import pickle
    # Load data (deserialize)
    with open(fn, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary


def run(name, alg, sizes=5, step=2, nopt=2 ** 10):
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
        "--repeat", required=False, default=1, help="Iterations inside measured region"
    )
    parser.add_argument(
        "--text", required=False, default="", help="Print with each result"
    )
    parser.add_argument("-d", type=int, default=3, help="Dimensions")
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or pure numpy",
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
    dims = int(args.d)

    output = {}
    output["name"] = name
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    if args.test:
        X, Y, p_D = gen_data(nopt, dims)
        pairwise_distance_python(X, Y, p_D)

        if args.usm is True:  # test usm feature
            X, Y, D_usm = gen_data_usm(nopt, dims)
            # pass usm input data to kernel
            alg(X, Y, D_usm)
            n_D = np.empty((nopt, nopt))
            D_usm.usm_data.copy_to_host(n_D.reshape((-1)).view("u1"))
        else:
            X, Y, n_D = gen_data(nopt, dims)
            # pass numpy generated data to kernel
            alg(X, Y, n_D)

        if np.allclose(n_D, p_D):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        if args.usm is True:
            X, Y, D = gen_data_usm(nopt, dims)
        else:
            X, Y, D = gen_data(nopt, dims)

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
        output["metrics"].append((nopt, mops, time))
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)
    f.close()
    f2.close()
    if repeat == 1:
        #print('base_pair_wise_graph.py:  repeat: ', repeat)
        resultsDict = {}
        resultsDict['X1'] = X[:1024]
        resultsDict['X2'] = Y[:1024]
        resultsDict['D'] = D[:1024,:1024]
        #print('X1.shape: ', resultsDict['X1'].shape)
        #print('X2.shape: ', resultsDict['X2'].shape)
        #print('D.shape: ', resultsDict['D'].shape)
        write_dictionary(resultsDict, 'resultsDict.pkl')