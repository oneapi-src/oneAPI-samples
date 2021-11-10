# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.random as rnd
import sys, json, os
import dpctl, dpctl.memory as dpmem, dpctl.tensor as dpt
from kmeans_python import kmeans_python
from generate_random_data import gen_rand_data

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


def gen_data_np(nopt):
    X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_rand_data(
        nopt, dtype=np.float32
    )
    return (X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint)


def gen_data_usm(nopt):
    X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_rand_data(
        nopt, dtype=np.float32
    )

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        X_usm = dpt.usm_ndarray(
            X.shape,
            dtype=X.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        arrayPclusters_usm = dpt.usm_ndarray(
            arrayPclusters.shape,
            dtype=arrayPclusters.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        arrayC_usm = dpt.usm_ndarray(
            arrayC.shape,
            dtype=arrayC.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        arrayCsum_usm = dpt.usm_ndarray(
            arrayCsum.shape,
            dtype=arrayCsum.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        arrayCnumpoint_usm = dpt.usm_ndarray(
            arrayCnumpoint.shape,
            dtype=arrayCnumpoint.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    X_usm.usm_data.copy_from_host(X.reshape((-1)).view("u1"))
    arrayPclusters_usm.usm_data.copy_from_host(arrayPclusters.view("u1"))
    arrayC_usm.usm_data.copy_from_host(arrayC.reshape((-1)).view("u1"))
    arrayCsum_usm.usm_data.copy_from_host(arrayCsum.reshape((-1)).view("u1"))
    arrayCnumpoint_usm.usm_data.copy_from_host(arrayCnumpoint.view("u1"))

    return (X_usm, arrayPclusters_usm, arrayC_usm, arrayCsum_usm, arrayCnumpoint_usm)


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


def run(name, alg, sizes=5, step=2, nopt=2 ** 13):
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

    f = open("perf_output.csv", "w")
    f2 = open("runtimes.csv", "w", 1)

    output = {}
    output["name"] = name
    output["sizes"] = sizes
    output["step"] = step
    output["repeat"] = repeat
    output["metrics"] = []

    if args.test:
        X, arrayPclusters_p, arrayC_p, arrayCsum_p, arrayCnumpoint_p = gen_data_np(nopt)
        kmeans_python(
            X,
            arrayPclusters_p,
            arrayC_p,
            arrayCsum_p,
            arrayCnumpoint_p,
            nopt,
            NUMBER_OF_CENTROIDS,
        )

        if args.usm is True:  # test usm feature
            # x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_usm = gen_data_usm(nopt)
            # alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED, result_usm)
            # result_n = np.empty(DEFAULT_NBINS-1, dtype=np.float64)
            # result_usm.usm_data.copy_to_host(result_n.view("u1"))
            (
                X,
                arrayPclusters,
                arrayC_usm,
                arrayCsum_usm,
                arrayCnumpoint_usm,
            ) = gen_data_usm(nopt)
            alg(
                X,
                arrayPclusters,
                arrayC_usm,
                arrayCsum_usm,
                arrayCnumpoint_usm,
                nopt,
                NUMBER_OF_CENTROIDS,
            )
            arrayC_n = np.empty((NUMBER_OF_CENTROIDS, 2), dtype=np.float32)
            arrayC_usm.usm_data.copy_to_host(arrayC_n.reshape((-1)).view("u1"))

            arrayCsum_n = np.empty((NUMBER_OF_CENTROIDS, 2), dtype=np.float32)
            arrayCsum_usm.usm_data.copy_to_host(arrayCsum_n.reshape((-1)).view("u1"))

            arrayCnumpoint_n = np.empty(NUMBER_OF_CENTROIDS, dtype=np.int32)
            arrayCnumpoint_usm.usm_data.copy_to_host(arrayCnumpoint_n.view("u1"))
        else:
            (
                X_n,
                arrayPclusters_n,
                arrayC_n,
                arrayCsum_n,
                arrayCnumpoint_n,
            ) = gen_data_np(nopt)

            # pass numpy generated data to kernel
            alg(
                X,
                arrayPclusters_n,
                arrayC_n,
                arrayCsum_n,
                arrayCnumpoint_n,
                nopt,
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
        if args.usm is True:
            X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_data_usm(nopt)
        else:
            X, arrayPclusters, arrayC, arrayCsum, arrayCnumpoint = gen_data_np(nopt)

        iterations = xrange(repeat)
        sys.stdout.flush()

        alg(
            X,
            arrayPclusters,
            arrayC,
            arrayCsum,
            arrayCnumpoint,
            nopt,
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
                nopt,
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
        resultsDict={}
        resultsDict['xC']=arrayC[:,0]
        resultsDict['yC']=arrayC[:,1]
        resultsDict['arrayP']=X
        resultsDict['arrayPclusters']=arrayPclusters
        #resultsDict['y']=arrayP[:,1]
        write_dictionary(resultsDict, 'resultsDict.pkl')