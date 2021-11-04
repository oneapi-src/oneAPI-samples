
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pair_wise_gpu
import numpy as np
import numba

# Naieve pairwise distance impl - take an array representing M points in N dimensions, and return the M x M matrix of Euclidean distances
@numba.njit(parallel=True, fastmath=True)
def pw_distance_kernel(X1, X2, D):
    # Size of imputs
    M = X1.shape[0]
    N = X2.shape[0]
    O = X1.shape[1]

    # Outermost parallel loop over the matrix X1
    for i in numba.prange(M):
        # Loop over the matrix X2
        for j in range(N):
            d = 0.0
            # Compute exclidean distance
            for k in range(O):
                tmp = X1[i, k] - X2[j, k]
                d += tmp * tmp
            # Write computed distance to distance matrix
            D[i, j] = np.sqrt(d)


def pw_distance(X1, X2, D):
    with dpctl.device_context(base_pair_wise_gpu.get_device_selector()):
        pw_distance_kernel(X1, X2, D)


base_pair_wise_gpu.run("Numba par_for", pw_distance)
