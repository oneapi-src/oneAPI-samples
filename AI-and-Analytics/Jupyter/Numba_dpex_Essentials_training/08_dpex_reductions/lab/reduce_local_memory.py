# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpctl
import dpctl.tensor as dpt
from numba import int32

import numba_dpex as ndpx


@ndpx.kernel
def sum_reduction_kernel(A, partial_sums):
    """
    The example demonstrates a reduction kernel implemented as a ``kernel``
    function.
    """
    local_id = ndpx.get_local_id(0)
    global_id = ndpx.get_global_id(0)
    group_size = ndpx.get_local_size(0)
    group_id = ndpx.get_group_id(0)

    local_sums = ndpx.local.array(64, int32)

    # Copy from global to local memory
    local_sums[local_id] = A[global_id]

    # Loop for computing local_sums : divide workgroup into 2 parts
    stride = group_size // 2
    while stride > 0:
        # Waiting for each 2x2 addition into given workgroup
        ndpx.barrier(ndpx.LOCAL_MEM_FENCE)

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

    partial_sums = dpt.zeros(nb_work_groups, dtype=A.dtype, device=A.device)

    gs = ndpx.Range(global_size)
    ls = ndpx.Range(work_group_size)
    sum_reduction_kernel[ndpx.NdRange(gs, ls)](A, partial_sums)

    final_sum = 0
    # calculate the final sum in HOST
    for i in range(nb_work_groups):
        final_sum += int(partial_sums[i])

    return final_sum


def test_sum_reduce():
    N = 1024
    device = dpctl.select_default_device()
    A = dpt.ones(N, dtype=dpt.int32, device=device)

    print("Running Device + Host reduction")

    actual = sum_reduce(A)
    expected = N

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert actual == expected

    print("Done...")


if __name__ == "__main__":
    test_sum_reduce()
