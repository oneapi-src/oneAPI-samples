##==============================================================
## Copyright © Intel Corporation
##
## SPDX-License-Identifier: Apache-2.0
## =============================================================
import dpctl
import numpy as np
from numba import float32

import numba_dpex as dpex


def private_memory():
    """
    This example demonstrates the usage of numba_dpex's `private.array`
    intrinsic function. The function is used to create a static array
    allocated on the devices private address space.
    """

    @dpex.kernel
    def private_memory_kernel(A):
        memory = dpex.private.array(shape=1, dtype=np.float32)
        i = dpex.get_global_id(0)

        # preload
        memory[0] = i
        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)  # local mem fence

        # memory will not hold correct deterministic result if it is not
        # private to each thread.
        A[i] = memory[0] * 2

    N = 4
    arr = np.zeros(N).astype(np.float32)
    orig = np.arange(N).astype(np.float32)

    # Use the environment variable SYCL_DEVICE_FILTER to change the default device.
    # See https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#sycl_device_filter.
    device = dpctl.select_default_device()
    print("Using device ...")
    device.print_device_info()

    with dpex.offload_to_sycl_device(device):
        private_memory_kernel[N, N](arr)

    #np.testing.assert_allclose(orig * 2, arr)
    # the output should be `orig[i] * 2, i.e. [0, 2, 4, ..]``
    print(arr)


def main():
    private_memory()
    print("Done...")

if __name__ == "__main__":
    main()
    
