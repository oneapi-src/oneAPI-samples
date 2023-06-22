##==============================================================
## Copyright © Intel Corporation
##
## SPDX-License-Identifier: Apache-2.0
## =============================================================

import dpctl
import numpy as np
from numba import float32

import numba_dpex as dpex


def no_arg_barrier_support():
    """
    This example demonstrates the usage of numba_dpex's ``barrier``
    intrinsic function. The ``barrier`` function is usable only inside
    a ``kernel`` and is equivalent to OpenCL's ``barrier`` function.
    """

    @dpex.kernel
    def twice(A):
        i = dpex.get_global_id(0)
        d = A[i]
        # no argument defaults to global mem fence
        dpex.barrier()
        A[i] = d * 2

    N = 10
    arr = np.arange(N).astype(np.float32)
    print(arr)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        twice[N, dpex.DEFAULT_LOCAL_SIZE](arr)

    # the output should be `arr * 2, i.e. [0, 2, 4, 6, ...]`
    print(arr)


def local_memory():
    """
    This example demonstrates the usage of numba-dpex's `local.array`
    intrinsic function. The function is used to create a static array
    allocated on the devices local address space.
    """
    blocksize = 10

    @dpex.kernel
    def reverse_array(A):
        lm = dpex.local.array(shape=10, dtype=float32)
        i = dpex.get_global_id(0)

        # preload
        lm[i] = A[i]
        # barrier local or global will both work as we only have one work group
        dpex.barrier(dpex.LOCAL_MEM_FENCE)  # local mem fence
        # write
        A[i] += lm[blocksize - 1 - i]

    arr = np.arange(blocksize).astype(np.float32)
    print(arr)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpctl.device_context(device):
        reverse_array[blocksize, dpex.DEFAULT_LOCAL_SIZE](arr)

    # the output should be `orig[::-1] + orig, i.e. [9, 9, 9, ...]``
    print(arr)


def main():
    no_arg_barrier_support()
    local_memory()

    print("Done...")


if __name__ == "__main__":
    main()
    
