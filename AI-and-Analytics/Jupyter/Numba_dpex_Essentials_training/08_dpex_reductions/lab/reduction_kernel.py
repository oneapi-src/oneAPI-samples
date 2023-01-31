##==============================================================
## Copyright © Intel Corporation
##
## SPDX-License-Identifier: Apache-2.0
## =============================================================

import math

import dpctl
import numpy as np

import numba_dpex as dpex
import timeit


@dpex.kernel
def sum_reduction_kernel(A, R, stride):
    i = dpex.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def sum_reduce(A):
    """Size of A should be power of two."""
    total = len(A)
    # max size will require half the size of A to store sum
    R = np.array(np.random.random(math.ceil(total / 2)), dtype=A.dtype)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    #print("Using device ...")
    #device.print_device_info()

    with dpctl.device_context(device):
        while total > 1:
            global_size = total // 2
            sum_reduction_kernel[global_size, dpex.DEFAULT_LOCAL_SIZE](
                A, R, global_size
            )
            total = total // 2

    return R[0]


def test_sum_reduce():
    # This test will only work for size = power of two
    N = 1024  

    A = np.array(np.random.random(N), dtype=np.float32)
    A_copy = A.copy()

    actual = sum_reduce(A)
    expected = A_copy.sum()

    #print("Actual:  ", actual)
    #print("Expected:", expected)   

    #print("Done...")
    
if __name__ == "__main__":
    t = timeit.Timer(lambda: test_sum_reduce())
    print("Time to calculate reduction",t.timeit(500),"seconds")
    
    
