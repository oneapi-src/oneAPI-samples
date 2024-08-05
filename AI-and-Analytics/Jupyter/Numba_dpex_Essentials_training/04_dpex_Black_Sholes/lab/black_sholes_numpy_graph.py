# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpnp as np
from numba_dpex import dpjit
import base_bs_erf_gpu

@dpjit
def black_scholes(nopt, price, strike, t, rate, volatility, call, put):
    mr = -rate
    sig_sig_two = volatility * volatility * 2

    P = price
    S = strike
    T = t

    a = np.log(P / S)
    b = T * mr

    z = T * sig_sig_two
    c = 0.25 * z
    y = np.true_divide(1.0, np.sqrt(z))

    w1 = (a - b + c) * y
    w2 = (a - b - c) * y

    d1 = 0.5 + 0.5 * np.erf(w1)
    d2 = 0.5 + 0.5 * np.erf(w2)

    Se = np.exp(b) * S

    call[:] = P * d1 - Se * d2
    put[:] = call - P + Se

# call the run function to setup input data and performance data infrastructure
base_bs_erf_gpu.run("Numba@jit-numpy", black_scholes)
