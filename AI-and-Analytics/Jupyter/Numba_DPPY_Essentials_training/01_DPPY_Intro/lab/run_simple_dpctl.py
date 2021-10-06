
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

from __future__ import print_function
from timeit import default_timer as time

import sys
import numpy as np
import numpy.testing as testing
from numba import dppl as dppy
import dpctl


@dppy.kernel
def data_parallel_sum(a, b, c):
    i = dppy.get_global_id(0)
    c[i] = a[i] + b[i]


def driver(a, b, c, global_size):
    print("A : ", a)
    print("B : ", b)
    data_parallel_sum[global_size, dppy.DEFAULT_LOCAL_SIZE](a, b, c)
    print("A + B = ")
    print("C ", c)
    testing.assert_equal(c, a + b)


def main():
    global_size = 10
    N = global_size
    print("N", N)

    a = np.array(np.random.random(N), dtype=np.float32)
    b = np.array(np.random.random(N), dtype=np.float32)
    c = np.ones_like(a)

    if dpctl.has_gpu_queues():
        print("\nScheduling on OpenCL GPU\n")
        with dpctl.device_context("opencl:gpu") as gpu_queue:
            driver(a, b, c, global_size)
    else:
        print("\nSkip scheduling on OpenCL GPU\n")
    if dpctl.has_gpu_queues(dpctl.backend_type.level_zero):
        print("\nScheduling on Level Zero GPU\n")
        with dpctl.device_context("level0:gpu") as gpu_queue:
            driver(a, b, c, global_size)
    else:
        print("\nSkip scheduling on Level Zero GPU\n")
    if dpctl.has_cpu_queues():
        print("\nScheduling on OpenCL CPU\n")
        with dpctl.device_context("opencl:cpu") as cpu_queue:
            driver(a, b, c, global_size)
    else:
        print("\nSkip scheduling on OpenCL CPU\n")
    print("Done...")


if __name__ == "__main__":
    main()
