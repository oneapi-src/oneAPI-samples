# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

from math import erf, exp, log, sqrt
import base_bs_erf_gpu
from numba import prange
from numba_dpex import dpjit


@dpjit
def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    for i in prange(nopt):
        P = price[i]
        S = strike[i]
        T = t[i]

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z
        y = 1.0 / sqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        r = P * d1 - Se * d2
        call[i] = r
        put[i] = r - P + Se

# call the run function to setup input data and performance data infrastructure
base_bs_erf_gpu.run("Numba@jit-loop-par", black_scholes)
