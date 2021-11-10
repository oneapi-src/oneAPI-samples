# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


from __future__ import print_function
import numpy as np
import sys, json, os
import dpctl, dpctl.tensor as dpt  # , dpctl.memory as dpmem
from bs_python import black_scholes_python

try:
    from numpy import erf

    numpy_ver += "-erf"
except:
    from scipy.special import erf

try:
    from numpy import invsqrt

    numpy_ver += "-invsqrt"
except:
    # from numba import jit
    invsqrt = lambda x: 1.0 / np.sqrt(x)
    # invsqrt = jit(['f8(f8)','f8[:](f8[:])'])(invsqrt)

try:
    import itimer as it

    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer

    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0), t1 - t0)

from generate_data_random import gen_rand_data

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2
TEST_ARRAY_LENGTH = 1024

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
    price, strike, t = gen_rand_data(nopt)
    return (
        price,
        strike,
        t,
        np.zeros(nopt, dtype=np.float64),
        -np.ones(nopt, dtype=np.float64),
    )


def gen_data_usm(nopt):
    # init numpy obj
    price_buf, strike_buf, t_buf = gen_rand_data(nopt)
    call_buf = np.zeros(nopt, dtype=np.float64)
    put_buf = -np.ones(nopt, dtype=np.float64)

    with dpctl.device_context(get_device_selector()) as gpu_queue:
        # init usmdevice memory
        # price_usm = dpmem.MemoryUSMDevice(nopt*np.dtype('f8').itemsize)
        # strike_usm = dpmem.MemoryUSMDevice(nopt*np.dtype('f8').itemsize)
        # t_usm = dpmem.MemoryUSMDevice(nopt*np.dtype('f8').itemsize)
        # call_usm = dpmem.MemoryUSMDevice(nopt*np.dtype('f8').itemsize)
        # put_usm = dpmem.MemoryUSMDevice(nopt*np.dtype('f8').itemsize)
        price_usm = dpt.usm_ndarray(
            price_buf.shape,
            dtype=price_buf.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        strike_usm = dpt.usm_ndarray(
            strike_buf.shape,
            dtype=strike_buf.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        t_usm = dpt.usm_ndarray(
            t_buf.shape,
            dtype=t_buf.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        call_usm = dpt.usm_ndarray(
            call_buf.shape,
            dtype=call_buf.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )
        put_usm = dpt.usm_ndarray(
            put_buf.shape,
            dtype=put_buf.dtype,
            buffer="device",
            buffer_ctor_kwargs={"queue": gpu_queue},
        )

    price_usm.usm_data.copy_from_host(price_buf.view("u1"))
    strike_usm.usm_data.copy_from_host(strike_buf.view("u1"))
    t_usm.usm_data.copy_from_host(t_buf.view("u1"))
    call_usm.usm_data.copy_from_host(call_buf.view("u1"))
    put_usm.usm_data.copy_from_host(put_buf.view("u1"))

    return (price_usm, strike_usm, t_usm, call_usm, put_usm)
    # return numpy obj with usmshared obj set to buffer
    # return(np.ndarray((nopt,), buffer=price_usm, dtype='f8'),
    #        np.ndarray((nopt,), buffer=strike_usm, dtype='f8'),
    #        np.ndarray((nopt,), buffer=t_usm, dtype='f8'),
    #        np.ndarray((nopt,), buffer=call_usm, dtype='f8'),
    #        np.ndarray((nopt,), buffer=put_usm, dtype='f8'))


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

# create input data, call blackscholes computation function (alg)
def run(name, alg, sizes=14, step=2, nopt=2 ** 15):
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
    kwargs = {}

    if args.test:
        price, strike, t, p_call, p_put = gen_data_np(nopt)
        black_scholes_python(
            nopt, price, strike, t, RISK_FREE, VOLATILITY, p_call, p_put
        )

        if args.usm is True:  # test usm feature
            price_usm, strike_usm, t_usm, call_usm, put_usm = gen_data_usm(nopt)
            # pass usm input data to kernel
            alg(
                nopt,
                price_usm,
                strike_usm,
                t_usm,
                RISK_FREE,
                VOLATILITY,
                call_usm,
                put_usm,
            )
            n_call = np.empty(nopt, dtype=np.float64)
            n_put = np.empty(nopt, dtype=np.float64)
            call_usm.usm_data.copy_to_host(n_call.view("u1"))
            put_usm.usm_data.copy_to_host(n_put.view("u1"))
        else:
            price_1, strike_1, t_1, n_call, n_put = gen_data_np(nopt)
            # pass numpy generated data to kernel
            alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, n_call, n_put)

        if np.allclose(n_call, p_call) and np.allclose(n_put, p_put):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        if args.usm is True:
            price, strike, t, call, put = gen_data_usm(nopt)
        else:
            price, strike, t, call, put = gen_data_np(nopt)

        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=" ", flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)  # warmup

        t0 = now()
        for _ in iterations:
            alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)

        mops, time = get_mops(t0, now(), nopt)

        # record performance data - mops, time
        print(
            "ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(
                name, nopt, mops * 2 * repeat, time
            ),
            flush=True,
        )
        output["metrics"].append((nopt, mops, time))
        f1.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output, open(args.json, "w"), indent=2, sort_keys=True)

    f1.close()
    f2.close()
    if repeat == 1:
        #print('base_bs_erf_graph.py:  repeat: ', repeat)
        resultsDict = {}
        resultsDict['price'] = price
        resultsDict['put'] = put
        resultsDict['call'] = call
        resultsDict['strike'] = strike 
#         print('price.shape: ', resultsDict['price'].shape)
#         print('strike.shape: ', resultsDict['strike'].shape)
#         print('put.shape: ', resultsDict['put'].shape)
#         print('call.shape: ', resultsDict['call'].shape)
        write_dictionary(resultsDict, 'resultsDict.pkl')