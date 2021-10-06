# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


from __future__ import print_function
import numpy as np
from random import seed, uniform
import sys, os
import dpctl, dpctl.memory as dpmem

try:
    import numpy.random_intel as rnd
    numpy_ver="Intel"
except:
    import numpy.random as rnd
    numpy_ver="regular"

try:
    from numpy import erf
    numpy_ver += "-erf"
except:
    from scipy.special import erf

try:
    from numpy import invsqrt
    numpy_ver += "-invsqrt"
except:
    #from numba import jit
    invsqrt = lambda x: 1.0/np.sqrt(x)
    #invsqrt = jit(['f8(f8)','f8[:](f8[:])'])(invsqrt)

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)


print("Using ", numpy_ver, " numpy ", np.__version__)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

SEED = 7777777
S0L = 10.0
S0H = 50.0
XL = 10.0
XH = 50.0
TL = 1.0
TH = 2.0
RISK_FREE = 0.1
VOLATILITY = 0.2
# RISK_FREE = np.float32(0.1)
# VOLATILITY = np.float32(0.2)
# C10 = np.float32(1.)
# C05 = np.float32(.5)
# C025 = np.float32(.25)
TEST_ARRAY_LENGTH = 1024

###############################################

def get_device_selector (is_gpu = True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if os.environ.get('SYCL_DEVICE_FILTER') is None or os.environ.get('SYCL_DEVICE_FILTER') == "opencl":
        return "opencl:" + device_selector

    if os.environ.get('SYCL_DEVICE_FILTER') == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get('SYCL_DEVICE_FILTER')

def gen_data_np(nopt):
    return (rnd.uniform(S0L, S0H, nopt),
            rnd.uniform(XL, XH, nopt),
            rnd.uniform(TL, TH, nopt),
            np.zeros(nopt, dtype=np.float64),
            -np.ones(nopt, dtype=np.float64))

def gen_data_usm(nopt):
    # init numpy obj
    price_buf = rnd.uniform(S0L, S0H, nopt)
    strike_buf = rnd.uniform(XL, XH, nopt)
    t_buf = rnd.uniform(TL, TH, nopt)
    call_buf = np.zeros(nopt, dtype=np.float64)
    put_buf  = -np.ones(nopt, dtype=np.float64)    

    with dpctl.device_context(get_device_selector()):    
        #copy numpy to usmshared
        price_usm = dpmem.MemoryUSMShared(nopt*np.dtype('f8').itemsize)
        strike_usm = dpmem.MemoryUSMShared(nopt*np.dtype('f8').itemsize)
        t_usm = dpmem.MemoryUSMShared(nopt*np.dtype('f8').itemsize)
        call_usm = dpmem.MemoryUSMShared(nopt*np.dtype('f8').itemsize)
        put_usm = dpmem.MemoryUSMShared(nopt*np.dtype('f8').itemsize)

        #return numpy obj with usmshared obj set to buffer
        price_usm.copy_from_host(price_buf.view("u1"))
        strike_usm.copy_from_host(strike_buf.view("u1"))
        t_usm.copy_from_host(t_buf.view("u1"))
        call_usm.copy_from_host(call_buf.view("u1"))
        put_usm.copy_from_host(put_buf.view("u1"))        

    return(np.ndarray((nopt,), buffer=price_usm, dtype='f8'),
           np.ndarray((nopt,), buffer=strike_usm, dtype='f8'),
           np.ndarray((nopt,), buffer=t_usm, dtype='f8'),
           np.ndarray((nopt,), buffer=call_usm, dtype='f8'),
           np.ndarray((nopt,), buffer=put_usm, dtype='f8'))

##############################################	

# create input data, call blackscholes computation function (alg)
def run(name, alg, sizes=14, step=2, nopt=2**15):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=100,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
	
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)

    rnd.seed(SEED)
    f1 = open("perf_output.csv",'w',1)
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        # generate input data
        if args.usm is True:
            price, strike, t, call, put = gen_data_usm(nopt)
        else:
            price, strike, t, call, put = gen_data_np(nopt)
            
        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        sys.stdout.flush()

        # call algorithm
        alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put) #warmup
            
        t0 = now()
        for _ in iterations:
            alg(nopt, price, strike, t, RISK_FREE, VOLATILITY, call, put)
            
        mops,time = get_mops(t0, now(), nopt)

        # record performance data - mops, time
        print("MOPS:", mops*2*repeat, "Time:", time, "Iters:", iterations)
        f1.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1

    f1.close()
    f2.close()