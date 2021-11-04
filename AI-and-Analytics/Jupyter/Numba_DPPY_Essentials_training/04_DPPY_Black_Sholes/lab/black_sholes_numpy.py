# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_bs_erf_gpu
import numba as nb
import numpy as np
from numpy import log, exp, sqrt
from math import erf

# Numba does know erf function from numpy or scipy
@nb.vectorize(nopython=True)
def nberf(x):
    return erf(x)

#blackscholes implemented using numpy function calls
@nb.jit(nopython=True,parallel=True,fastmath=True)
def black_scholes_kernel( nopt, price, strike, t, rate, vol, call, put ):
    mr = -rate
    sig_sig_two = vol * vol * 2

    P = price
    S = strike
    T = t

    a = log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = 1./sqrt(z)

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * nberf(w1)
    d2 = 0.5 + 0.5 * nberf(w2)

    Se = exp(b) * S

    r =  P * d1 - Se * d2
    call[:] = r  # temporary `r` is necessary for faster `put` computation
    put[:] = r - P + Se

def black_scholes(nopt, price, strike, t, rate, vol, call, put):
    # offload blackscholes computation to GPU (toggle level0 or opencl driver).
    with dpctl.device_context(base_bs_erf_gpu.get_device_selector()):
        black_scholes_kernel( nopt, price, strike, t, rate, vol, call, put )

# call the run function to setup input data and performance data infrastructure
base_bs_erf_gpu.run("Numba@jit-numpy", black_scholes)
