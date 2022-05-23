
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import dpctl
import base_pair_wise_gpu
import numpy as np
import numba


# Pairwise Numpy implementation using the equation (a-b)^2 = a^2 + b^2 - 2*a*b
@numba.njit(parallel=True, fastmath=True)
def pw_distance_kernel(X1, X2, D):
    # return np.sqrt((np.square(X1 - X2.reshape((X2.shape[0],1,X2.shape[1])))).sum(axis=2))

    # Computing the first two terms (X1^2 and X2^2) of the Euclidean distance equation
    x1 = np.sum(np.square(X1), axis=1)
    x2 = np.sum(np.square(X2), axis=1)

    # Comnpute third term in equation
    D = -2 * np.dot(X1, X2.T)
    x3 = x1.reshape(x1.size, 1)
    D = D + x3  # x1[:,None] Not supported by Numba
    D = D + x2

    # Compute square root for euclidean distance
    D = np.sqrt(D)


def pw_distance(X1, X2, D):
    with dpctl.device_context(base_pair_wise_gpu.get_device_selector()):
        pw_distance_kernel(X1, X2, D)


base_pair_wise_gpu.run("Numba Numpy", pw_distance) 
