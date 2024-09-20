# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math

import dpnp as np

import numba_dpex as ndpx


@ndpx.kernel
def sum_reduction_kernel(A, R, stride):
    i = ndpx.get_global_id(0)
    # sum two element
    R[i] = A[i] + A[i + stride]
    # store the sum to be used in nex iteration
    A[i] = R[i]


def sum_reduce(A):
    """Size of A should be power of two."""
    total = len(A)
    # max size will require half the size of A to store sum
    R = np.array(np.random.random(math.floor(total / 2)), dtype=A.dtype)

    while total > 1:
        global_size = math.floor(total // 2)
        total = total - global_size
        sum_reduction_kernel[ndpx.Range(global_size)](A, R, total)

    return R[0]


def test_sum_reduce():
    N = 2048

    A = np.arange(N, dtype=np.float32)
    A_copy = np.arange(N, dtype=np.float32)

    actual = sum_reduce(A)
    expected = A_copy.sum()

    print("Actual:  ", actual)
    print("Expected:", expected)

    assert expected - actual < 1e-2

    print("Done...")


if __name__ == "__main__":
    test_sum_reduce()
    
