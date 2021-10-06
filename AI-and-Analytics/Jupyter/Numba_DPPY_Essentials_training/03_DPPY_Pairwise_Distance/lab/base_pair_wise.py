# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT


import numpy as np
import sys
import numpy.random as rnd

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
SEED = 7777777
# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################

def gen_data(nopt,dims):
    return (
        rnd.random((nopt, dims)),
        rnd.random((nopt, dims)),
        np.empty((nopt, nopt))
    )

##############################################	

def run(name, alg, sizes=5, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=100,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('-d', type=int, default=3, help='Dimensions')
    
    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)
    dims = int(args.d)
    
    rnd.seed(SEED)
    f=open("perf_output.csv",'w',1)
    f2 = open("runtimes.csv",'w',1)
    
    for i in xrange(sizes):
        X,Y,D = gen_data(nopt,dims)
        iterations = xrange(repeat)
        print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        sys.stdout.flush()

        alg(X,Y,D) #warmup
        t0 = now()
        for _ in iterations:
            alg(X,Y,D)

        mops,time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print(str(nopt) + "," + str(time) + "\n")
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    f.close()
    f2.close()