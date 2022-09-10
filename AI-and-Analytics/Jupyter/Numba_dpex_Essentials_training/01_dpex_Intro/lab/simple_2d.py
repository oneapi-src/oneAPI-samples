#! /usr/bin/env python
# Copyright 2020, 2021 Intel Corporation
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

import dpctl
import numba_dppy as dppy
import numpy as np


@dppy.kernel
def data_parallel_sum(a, b, c):
    """
    A two-dimensional vector addition example using the ``kernel`` decorator.
    """
    i = dppy.get_global_id(0)
    j = dppy.get_global_id(1)
    c[i, j] = a[i, j] + b[i, j]


def driver(a, b, c, global_size):
    print("before A: ", a)
    print("before B: ", b)
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
    print("after  C : ", c)


def main():
    # Array dimensions
    X = 8
    Y = 8
    global_size = X, Y

    a = np.arange(X * Y, dtype=np.float32).reshape(X, Y)
    b = np.array(np.random.random(X * Y), dtype=np.float32).reshape(X, Y)
    c = np.ones_like(a).reshape(X, Y)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        driver(a, b, c, global_size)

    print(c)

    print("Done...")


if __name__ == "__main__":
    main()
