# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import sys, os, json, datetime

# import numpy.random_intel as rnd
import numpy.random as rnd
import dpctl
import dpctl.tensor as dpt

from l2_distance_python import l2_distance_python
from generate_data_l2 import gen_data
from generate_data_l2 import SEED

from device_selector import get_device_selector

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


def gen_data_usm(nopt, dims):
    x, y = gen_data(nopt, dims, np.float32)
    distance = np.asarray([0.0]).astype(np.float32)

    with dpctl.device_context(get_device_selector(is_gpu=True)) as gpu_queue:
        x_usm = dpt.usm_ndarray(
            x.shape,
            dtype=x.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        y_usm = dpt.usm_ndarray(
            y.shape,
            dtype=y.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        distance_usm = dpt.usm_ndarray(
            distance.shape,
            dtype=distance.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    x_usm.usm_data.copy_from_host(x.reshape((-1)).view("|u1"))
    y_usm.usm_data.copy_from_host(y.reshape((-1)).view("|u1"))
    distance_usm.usm_data.copy_from_host(distance.reshape((-1)).view("|u1"))

    return x_usm, y_usm, distance_usm


##############################################


def run(name, alg, sizes=2, step=2, nopt=2 ** 20):
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
    parser.add_argument(
        "--json",
        required=False,
        default=__file__.replace("py", "json"),
        help="output json data filename",
    )
    parser.add_argument("-d", type=int, default=1, help="Dimensions")
    parser.add_argument(
        "--test", required=False, action="store_true", help="Validation"
    )
    parser.add_argument(
        "--usm",
        required=False,
        action="store_true",
        help="Use USM Shared or pure numpy",
    )

    args = parser.parse_args()
    sizes = int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat = int(args.repeat)
    dims = int(args.d)

    f = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    output = {}
    output["name"] = name
    output["datetime"] = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"
    )
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["dims"] = dims
    output["randseed"] = SEED
    output["metrics"] = []

    times = np.empty(repeat)
    if args.test:
        X, Y = gen_data(nopt, dims, np.float32)
        p_dis = l2_distance_python(X, Y)

        if args.usm is True:  # test usm feature
            X_usm, Y_usm, distance = gen_data_usm(nopt, dims)
            n_dis = alg(X_usm, Y_usm, distance)

        else:
            distance = np.asarray([0.0]).astype(np.float32)
            n_dis = alg(X, Y, distance)

        # RMS error grows proportional to sqrt(n)
        # absolute(a - b) <= (atol + rtol * absolute(b))
        #if np.allclose(n_dis, p_dis, rtol=1e-05 * np.sqrt(nopt)):
            #print("Test succeeded. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")
        #else:
            #print("Test failed. Python dis: ", p_dis, " Numba dis: ", n_dis, "\n")

    for _ in xrange(sizes):
        if args.usm is True:
            X, Y, distance = gen_data_usm(nopt, dims)
        else:
            X, Y = gen_data(nopt, dims, np.float32)
            distance = np.asarray([0.0]).astype(np.float32)

        iterations = xrange(repeat)
        # print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        sys.stdout.flush()

        n_dis = alg(X, Y, distance)  # warmup

        for i in iterations:
            distance = np.asarray([0.0]).astype(np.float32)

            X, Y = gen_data(nopt, dims, np.float32)
            p_dis = l2_distance_python(X, Y)

            t0 = default_timer()
            n_dis = alg(X, Y, distance)
            t1 = default_timer()

            times[i] = t1 - t0            

        time = np.median(times)

        print(
            "ERF: {:15s} | Size: {:10d} | TIME: {:10.6f}".format(name, nopt, time),
            flush=True,
        )
        output["metrics"].append((nopt, 0, time))  # zero placeholder for mops
        # f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")

        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)
    f.close()
    f2.close()