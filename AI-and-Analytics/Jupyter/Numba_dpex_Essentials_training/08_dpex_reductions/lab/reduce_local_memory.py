##==============================================================
## Copyright © Intel Corporation
##
## SPDX-License-Identifier: Apache-2.0
## =============================================================

import dpctl
import numpy as np
from numba import int32

import numba_dpex as dpex
import timeit


@dpex.kernel
def sum_reduction_kernel(A, partial_sums):
    """
    The example demonstrates a reduction kernel implemented as a ``kernel``
    function.
    """
    local_id = dpex.get_local_id(0)
    global_id = dpex.get_global_id(0)
    group_size = dpex.get_local_size(0)
    group_id = dpex.get_group_id(0)

    local_sums = dpex.local.array(64, int32)

    # Copy from global to local memory
    local_sums[local_id] = A[global_id]

    # Loop for computing local_sums : divide workgroup into 2 parts
    stride = group_size // 2
    while stride > 0:
        # Waiting for each 2x2 addition into given workgroup
        dpex.barrier(dpex.CLK_LOCAL_MEM_FENCE)

        # Add elements 2 by 2 between local_id and local_id + stride
        if local_id < stride:
            local_sums[local_id] += local_sums[local_id + stride]

        stride >>= 1

    if local_id == 0:
        partial_sums[group_id] = local_sums[0]


def sum_reduce(A):
    global_size = len(A)
    work_group_size = 64
    # nb_work_groups have to be even for this implementation
    nb_work_groups = global_size // work_group_size

    partial_sums = np.zeros(nb_work_groups).astype(A.dtype)
    
    device = dpctl.select_default_device()
    #print("Using device ...")
    #device.print_device_info()

    with dpctl.device_context(device):
        sum_reduction_kernel[global_size, work_group_size](A, partial_sums)

    final_sum = 0
    # calculate the final sum in HOST
    for i in range(nb_work_groups):
        final_sum += partial_sums[i]

    return final_sum


def test_sum_reduce():
    N = 1024
    #A = np.ones(N).astype(np.int32)
    A = np.arange(N)
    #A = np.array(np.random.random(N), dtype=np.float32)

    #print("Running Device + Host reduction")

    actual = sum_reduce(A)
    #expected = N

    #print("Actual:  ", actual)
    #print("Expected:", expected)

    #assert actual == expected

    #print("Done...")


if __name__ == "__main__":
    #test_sum_reduce()
    t = timeit.Timer(lambda: test_sum_reduce())    
    print("Time to calculate the sum in Local memory",t.timeit(500),"seconds")
    
