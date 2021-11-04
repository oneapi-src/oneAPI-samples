# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os, json
import numpy as np
import dpctl, dpctl.tensor as dpt

from gpairs_python import gpairs_python
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
DEFAULT_NBINS = 20

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


def gen_data_np(npoints, dtype=np.float32):
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED = gen_rand_data(
        npoints, dtype
    )
    result = np.zeros_like(DEFAULT_RBINS_SQUARED)[:-1].astype(dtype)
    return (x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)


def gen_data_usm(npoints):
    # init numpy obj
    x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result = gen_data_np(npoints)

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        # init usmdevice memory
        x1_usm = dpt.usm_ndarray(
            x1.shape,
            dtype=x1.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        y1_usm = dpt.usm_ndarray(
            y1.shape,
            dtype=y1.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        z1_usm = dpt.usm_ndarray(
            z1.shape,
            dtype=z1.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        w1_usm = dpt.usm_ndarray(
            w1.shape,
            dtype=w1.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        x2_usm = dpt.usm_ndarray(
            x2.shape,
            dtype=x2.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        y2_usm = dpt.usm_ndarray(
            y2.shape,
            dtype=y2.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        z2_usm = dpt.usm_ndarray(
            z2.shape,
            dtype=z2.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        w2_usm = dpt.usm_ndarray(
            w2.shape,
            dtype=w2.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        DEFAULT_RBINS_SQUARED_usm = dpt.usm_ndarray(
            DEFAULT_RBINS_SQUARED.shape,
            dtype=DEFAULT_RBINS_SQUARED.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        result_usm = dpt.usm_ndarray(
            result.shape,
            dtype=result.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    x1_usm.usm_data.copy_from_host(x1.view("u1"))
    y1_usm.usm_data.copy_from_host(y1.view("u1"))
    z1_usm.usm_data.copy_from_host(z1.view("u1"))
    w1_usm.usm_data.copy_from_host(w1.view("u1"))
    x2_usm.usm_data.copy_from_host(x2.view("u1"))
    y2_usm.usm_data.copy_from_host(y2.view("u1"))
    z2_usm.usm_data.copy_from_host(z2.view("u1"))
    w2_usm.usm_data.copy_from_host(w2.view("u1"))
    DEFAULT_RBINS_SQUARED_usm.usm_data.copy_from_host(DEFAULT_RBINS_SQUARED.view("u1"))
    result_usm.usm_data.copy_from_host(result.view("u1"))

    return (
        x1_usm,
        y1_usm,
        z1_usm,
        w1_usm,
        x2_usm,
        y2_usm,
        z2_usm,
        w2_usm,
        DEFAULT_RBINS_SQUARED_usm,
        result_usm,
    )


##############################################


def run(name, alg, sizes=5, step=2, nopt=2 ** 16):
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

    output = {}
    output["name"] = name
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    if args.test:
        x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_p = gen_data_np(
            nopt
        )
        gpairs_python(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_p)

        if args.usm is True:  # test usm feature
            (
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result_usm,
            ) = gen_data_usm(nopt)
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_usm)
            result_n = np.empty(DEFAULT_NBINS - 1, dtype=np.float32)
            result_usm.usm_data.copy_to_host(result_n.view("u1"))
        else:
            (
                x1_n,
                y1_n,
                z1_n,
                w1_n,
                x2_n,
                y2_n,
                z2_n,
                w2_n,
                DEFAULT_RBINS_SQUARED,
                result_n,
            ) = gen_data_np(nopt)

            # pass numpy generated data to kernel
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_n)

        if np.allclose(result_p, result_n):
            print("Test succeeded\n")
        else:
            print(
                "Test failed\n",
                "Python result: ",
                result_p,
                "\n numba result:",
                result_n,
            )
        return

    f = open("perf_output.csv", "w")
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        if args.usm is True:
            (
                x1,
                y1,
                z1,
                w1,
                x2,
                y2,
                z2,
                w2,
                DEFAULT_RBINS_SQUARED,
                result,
            ) = gen_data_usm(nopt)
        else:
            x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result = gen_data_np(
                nopt
            )
        iterations = xrange(repeat)

        alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)  # warmup
        t0 = now()
        for _ in iterations:
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result)

        mops, time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * 2 * repeat, time
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