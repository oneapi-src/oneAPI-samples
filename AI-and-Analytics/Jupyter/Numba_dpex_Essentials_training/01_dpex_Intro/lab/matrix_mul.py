#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx

#***Step1: Uncomment the following lines to enable the dpex.kernel decorator***
@ndpx.kernel
def gemm(a, b, c):
    """
    A basic DGEMM implemented as a ``kernel`` function.
    """
    i = ndpx.get_global_id(0)
    j = ndpx.get_global_id(1)
    if i >= c.shape[0] or j >= c.shape[1]:
        return
    c[i, j] = 0
    for k in range(c.shape[0]):
        c[i, j] += a[i, k] * b[k, j]


# Array dimensions
X = 1024
Y = 16
global_size = X, X

griddim = ndpx.Range(X, X)
blockdim = ndpx.Range(Y, Y)


def driver(a, b, c):
    # Invoke the kernel
    gemm[ndpx.NdRange(griddim, blockdim)](a, b, c)


def main():
    a = np.arange(X * X, dtype=np.float32).reshape(X, X)
    b = np.array(np.random.random(X * X), dtype=np.float32).reshape(X, X)
    
    #***Step2: Uncomment the following lines to set the device context and target a GPU***
    device = dpctl.select_default_device()
    a_dpt = dpt.arange(X * X, dtype=dpt.float32, device=device)
    a_dpt = dpt.reshape(a_dpt, (X, X))
    b_dpt = dpt.asarray(b, dtype=dpt.float32, device=device)
    b_dpt = dpt.reshape(b_dpt, (X, X))
    c_dpt = dpt.ones_like(a_dpt)
    c_dpt = dpt.reshape(c_dpt, (X, X))

    print("Using device ...")
    device.print_device_info()

    driver(a_dpt, b_dpt, c_dpt)
    c_out = dpt.asnumpy(c_dpt)

    # Host compute using standard NumPy
    Amat = np.matrix(a)
    Bmat = np.matrix(b)
    Cans = Amat * Bmat

    # Check result
    assert np.allclose(c_out, Cans)

    print("Done...")


if __name__ == "__main__":
    main()
