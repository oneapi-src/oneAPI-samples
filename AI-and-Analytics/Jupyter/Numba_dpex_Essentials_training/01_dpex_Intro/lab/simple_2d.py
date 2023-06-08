#! /usr/bin/env python

# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
import numpy as np

import numba_dpex as ndpx


@ndpx.kernel
def data_parallel_sum(a, b, c):
    """
    A two-dimensional vector addition example using the ``kernel`` decorator.
    """
    i = ndpx.get_global_id(0)
    j = ndpx.get_global_id(1)
    c[i, j] = a[i, j] + b[i, j]


def driver(a, b, c, global_size):
    data_parallel_sum[global_size](a, b, c)


def main():
    # Array dimensions
    X = 8
    Y = 8
    global_size = ndpx.Range(X, Y)

    a = np.arange(X * Y, dtype=np.float32).reshape(X, Y)
    b = np.arange(X * Y, dtype=np.float32).reshape(X, Y)
    c = np.empty_like(a).reshape(X, Y)

    c = a + b

    device = dpctl.select_default_device()
    a_dpt = dpt.arange(X * Y, dtype=dpt.float32, device=device)
    a_dpt = dpt.reshape(a_dpt, (X, Y))
    b_dpt = dpt.arange(X * Y, dtype=dpt.float32, device=device)
    b_dpt = dpt.reshape(b_dpt, (X, Y))
    c_dpt = dpt.empty_like(a_dpt)
    c_dpt = dpt.reshape(c_dpt, (X, Y))

    print("Using device ...")
    device.print_device_info()

    print("Running kernel ...")
    driver(a_dpt, b_dpt, c_dpt, global_size)
    c_out = dpt.asnumpy(c_dpt)
    assert np.allclose(c, c_out)

    print("Done...")


if __name__ == "__main__":
    main()
