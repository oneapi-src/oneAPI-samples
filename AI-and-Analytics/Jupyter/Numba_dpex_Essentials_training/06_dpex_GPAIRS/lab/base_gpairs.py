# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
import os,json
import numpy as np
import numpy.random as rnd
try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

SEED = 7777777
DEFAULT_SEED=SEED
DEFAULT_NBINS = 20
DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50
DEFAULT_RBINS = np.logspace(
    np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
        np.float32)
DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(np.float32)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################

def random_weighted_points(n, Lbox, seed=DEFAULT_SEED):
    """
    """
    rng = rnd.RandomState(seed)
    data = rng.uniform(0, 1, n*4)
    x, y, z, w = (
        data[:n]*Lbox, data[n:2*n]*Lbox, data[2*n:3*n]*Lbox, data[3*n:])
    return (
        x.astype(np.float32), y.astype(np.float32), z.astype(np.float32),
        w.astype(np.float32))

def gen_data(npoints):
    Lbox = 500.
    n1 = npoints
    n2 = npoints
    x1, y1, z1, w1 = random_weighted_points(n1, Lbox, 0)
    x2, y2, z2, w2 = random_weighted_points(n2, Lbox, 1)

    return (
        x1, y1, z1, w1, x2, y2, z2, w2
    )

def copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2):
    device_env = ocldrv.runtime.get_gpu_device()
    d_x1 = device_env.copy_array_to_device(x1.astype(np.float32))
    d_y1 = device_env.copy_array_to_device(y1.astype(np.float32))
    d_z1 = device_env.copy_array_to_device(z1.astype(np.float32))
    d_w1 = device_env.copy_array_to_device(w1.astype(np.float32))

    d_x2 = device_env.copy_array_to_device(x2.astype(np.float32))
    d_y2 = device_env.copy_array_to_device(y2.astype(np.float32))
    d_z2 = device_env.copy_array_to_device(z2.astype(np.float32))
    d_w2 = device_env.copy_array_to_device(w2.astype(np.float32))

    d_rbins_squared = device_env.copy_array_to_device(
        DEFAULT_RBINS_SQUARED.astype(np.float32))

    return (
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared
    )

def copy_d2h(d_result):
    device_env.copy_array_from_device(d_result)

##############################################

def run(name, alg, sizes=10, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")

    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)

    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []

    rnd.seed(SEED)

    f=open("perf_output.csv",'w')
    f2 = open("runtimes.csv",'w',1)

    for i in xrange(sizes):
        x1, y1, z1, w1, x2, y2, z2, w2 = gen_data(nopt)
        #d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared = copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2)
        iterations = xrange(repeat)
        #print("ERF: {}: Size: {}".format(name, nopt), end=' ', flush=True)
        #sys.stdout.flush()

        alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED) #warmup
        t0 = now()
        for _ in iterations:
            #t1 = now()
            alg(x1, y1, z1, w1, x2, y2, z2, w2, DEFAULT_RBINS_SQUARED)
            #print("Time:", now()-t1)

        mops,time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print("ERF: {:15s} | Size: {:10d} | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, mops*2*repeat,time),flush=True)
        output['metrics'].append((nopt,mops,time))
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
    f.close()
    f2.close()