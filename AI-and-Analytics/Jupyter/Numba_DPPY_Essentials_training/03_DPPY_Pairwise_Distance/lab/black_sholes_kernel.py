# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
import time
import numba_dppy, numba_dppy as dppy
import unittest
import dpctl


RISKFREE = 0.02
VOLATILITY = 0.30

A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


OPT_N = 400
iterations = 2

stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)
callResult = np.zeros(OPT_N)
putResult = -np.ones(OPT_N)


@dppy.kernel
def black_scholes_dppy(callResult, putResult, S, X, T, R, V):
    i = dppy.get_global_id(0)
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT

    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d1))
    cndd1 = (
        RSQRT2PI
        * math.exp(-0.5 * d1 * d1)
        * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    if d1 > 0:
        cndd1 = 1.0 - cndd1

    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d2))
    cndd2 = (
        RSQRT2PI
        * math.exp(-0.5 * d2 * d2)
        * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    if d2 > 0:
        cndd2 = 1.0 - cndd2

    expRT = math.exp((-1.0 * R) * T[i])
    callResult[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    putResult[i] = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)


blockdim = 512, 1
griddim = int(math.ceil(float(OPT_N) / blockdim[0])), 1

with dpctl.device_context("opencl:gpu") as gpu_queue:
    time1 = time.time()
    for i in range(iterations):
        black_scholes_dppy[blockdim, griddim](
            callResult,
            putResult,
            stockPrice,
            optionStrike,
            optionYears,
            RISKFREE,
            VOLATILITY,
        )

print("callResult : ", callResult)
print("putResult : ", putResult)
