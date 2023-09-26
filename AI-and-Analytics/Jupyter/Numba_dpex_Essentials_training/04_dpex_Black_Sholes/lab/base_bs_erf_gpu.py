# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import sys
import dpnp as np
import numpy
from bs_python import black_scholes_python
from generate_data_random import gen_rand_data
import dpctl

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

RISK_FREE = 0.1
VOLATILITY = 0.2
    
###############################################

def gen_data_np(nopt):
    price, strike, t = gen_rand_data(nopt)
    call = numpy.zeros(nopt, dtype=np.float64)
    put = numpy.ones(nopt, dtype=np.float64)
    return (
        price,
        strike,
        t,
        call,
        put,
    )

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
     price, strike, t, call, put = gen_data_np(nopt)

     #convert to dpnp
     return (to_dpnp(price), to_dpnp(strike), to_dpnp(t), to_dpnp(call), to_dpnp(put))
     
##############################################

# create input data, call blackscholes computation function (alg)
def run(name, alg, sizes=10, step=2, nopt=2**19):
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

    dpctl.SyclDevice("gpu")

    if args.test:
        price, strike, t, p_call, p_put = gen_data_np(nopt)
        black_scholes_python(
            nopt, price, strike, t, RISK_FREE, VOLATILITY, p_call, p_put
        )

        n_price, n_strike, n_t, n_call, n_put = gen_data_dpnp(nopt)
        # pass numpy generated data to kernel
        alg(nopt, n_price, n_strike, n_t, RISK_FREE, VOLATILITY, n_call, n_put)

        if numpy.allclose(to_numpy(n_call), p_call) and numpy.allclose(to_numpy(n_put), p_put):
            print("Test succeeded\n")
        else:
            print("Test failed\n")
        return

    f1 = open("perf_output.csv", "w", 1)
    f2 = open("runtimes.csv", "w", 1)

    for i in xrange(sizes):
        # generate input data
        price, strike, t, call, put = gen_data_dpnp(nopt)

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
        f1.write(str(nopt) + "," + str(mops * 2 * repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f1.close()
    f2.close()