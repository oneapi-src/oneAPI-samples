
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
import numba_dppy as dppy
import dpctl


@dppy.kernel
def sum_reduction_kernel(A, R, stride):
    i = dppy.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def get_context():
    if dpctl.has_gpu_queues():
        return "opencl:gpu"
    elif dpctl.has_cpu_queues():
        return "opencl:cpu"
    else:
        raise RuntimeError("No device found")


def sum_reduce(A):
    """Size of A should be power of two."""
    total = len(A)
    # max size will require half the size of A to store sum
    R = np.array(np.random.random(math.ceil(total / 2)), dtype=A.dtype)

    context = get_context()
    with dpctl.device_context(context):
        while total > 1:
            global_size = total // 2
            sum_reduction_kernel[global_size, dppy.DEFAULT_LOCAL_SIZE](
                A, R, global_size
            )
            total = total // 2

    return R[0]


def test_sum_reduce():
    # This test will only work for size = power of two
    N = 2048
    assert N % 2 == 0

    A = np.array(np.random.random(N), dtype=np.float32)
    A_copy = A.copy()

    actual = sum_reduce(A)
    expected = A_copy.sum()

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert expected - actual < 1e-2


if __name__ == "__main__":
    test_sum_reduce()
