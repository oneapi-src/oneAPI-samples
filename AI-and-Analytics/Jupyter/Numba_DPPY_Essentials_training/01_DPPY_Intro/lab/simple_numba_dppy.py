
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
import numba_dppy as dppy
import dpctl

"""
    A `func` is a device callable function that can be invoked from
    `kernel` and other `func` functions.
    """
@dppy.func
def a_device_function(a):
    return a + 1


@dppy.func
def an_another_device_function(a):
    return a_device_function(a)


@dppy.kernel
def a_kernel_function(a, b):
    i = dppy.get_global_id(0)
    b[i] = an_another_device_function(a[i])


def driver(a, b, N):
    print(b)
    print("--------")
    a_kernel_function[N, dppy.DEFAULT_LOCAL_SIZE](a, b)
    print(b)


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)

    context = get_context()

    print("Device Context:", context)

    with dpctl.device_context(context):
        driver(a, b, N)


if __name__ == "__main__":
    main()
