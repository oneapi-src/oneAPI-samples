# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import numpy.random as rnd
import sys,os
import dpctl, dpctl.memory as dpmem

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

SEED = 7777777
XL = 1.0
XH = 5.0
dims = 2
NUMBER_OF_CENTROIDS = 10

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
    return (
        rnd.uniform(XL, XH, (nopt, dims)),
        np.ones(nopt, dtype=np.int32),
        np.ones((NUMBER_OF_CENTROIDS, 2), dtype=np.float64),
        np.ones((NUMBER_OF_CENTROIDS, 2), dtype=np.float64),
        np.ones(NUMBER_OF_CENTROIDS, dtype=np.int32)        
    )

def gen_data_usm(nopt):
    X_buf = rnd.uniform(XL, XH, (nopt, dims))
    arrayPclusters_buf = np.ones(nopt, dtype=np.int32)
    arrayC_buf = np.ones((NUMBER_OF_CENTROIDS, dims), dtype=np.float64)
    arrayCsum_buf = np.ones((NUMBER_OF_CENTROIDS, dims), dtype=np.float64)
    arrayCnumpoint_buf = np.ones(NUMBER_OF_CENTROIDS, dtype=np.int32)

    with dpctl.device_context(get_device_selector()):
        X_usm = dpmem.MemoryUSMShared(nopt*dims*np.dtype('f8').itemsize)
        arrayPclusters_usm = dpmem.MemoryUSMShared(nopt*np.dtype('i4').itemsize)
        arrayC_usm = dpmem.MemoryUSMShared(NUMBER_OF_CENTROIDS*dims*np.dtype('f8').itemsize)
        arrayCsum_usm = dpmem.MemoryUSMShared(NUMBER_OF_CENTROIDS*dims*np.dtype('f8').itemsize)
        arrayCnumpoint_usm = dpmem.MemoryUSMShared(NUMBER_OF_CENTROIDS*np.dtype('i4').itemsize)

        X_usm.copy_from_host(X_buf.reshape((-1)).view("u1"))
        arrayPclusters_usm.copy_from_host(arrayPclusters_buf.view("u1"))
        arrayC_usm.copy_from_host(arrayC_buf.reshape((-1)).view("u1"))
        arrayCsum_usm.copy_from_host(arrayCsum_buf.reshape((-1)).view("u1"))
        arrayCnumpoint_usm.copy_from_host(arrayCnumpoint_buf.view("u1"))
    
    return (np.ndarray((nopt,dims), buffer=X_usm, dtype='f8'),
            np.ndarray((nopt,), buffer=arrayPclusters_usm, dtype='i4'),
            np.ndarray((NUMBER_OF_CENTROIDS, dims), buffer=arrayC_usm, dtype='f8'),
            np.ndarray((NUMBER_OF_CENTROIDS, dims), buffer=arrayCsum_usm, dtype='f8'),
            np.ndarray((NUMBER_OF_CENTROIDS,), buffer=arrayCnumpoint_usm, dtype='i4'))

##############################################	

def run(name, alg, sizes=10, step=2, nopt=2**13):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('--usm',   required=False, action='store_true',  help="Use USM Shared or pure numpy")
    
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)

    rnd.seed(SEED)
    f=open("perf_output.csv",'w')
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        if args.usm is True:
            X,arrayPclusters,arrayC,arrayCsum,arrayCnumpoint = gen_data_usm(nopt)
        else:
            X,arrayPclusters,arrayC,arrayCsum,arrayCnumpoint = gen_data_np(nopt)
            
        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        sys.stdout.flush()

        alg(X, arrayPclusters,arrayC,arrayCsum,arrayCnumpoint, nopt, NUMBER_OF_CENTROIDS) #warmup
        t0 = now()
        for _ in iterations:
            alg(X, arrayPclusters,arrayC,arrayCsum,arrayCnumpoint, nopt, NUMBER_OF_CENTROIDS)

        mops,time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print("Time:", str(time))
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
    f2.close()