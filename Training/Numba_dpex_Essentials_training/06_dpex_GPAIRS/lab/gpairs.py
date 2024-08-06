
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_gpairs
import numpy as np
from gaussian_weighted_pair_counts import count_weighted_pairs_3d_cpu

def run_gpairs(x1, y1, z1, w1, x2, y2, z2, w2, d_rbins_squared):
    x1 = x1.astype(np.float32)
    y1 = y1.astype(np.float32)
    z1 = z1.astype(np.float32)
    w1 = w1.astype(np.float32)
    x2 = x2.astype(np.float32)
    y2 = y2.astype(np.float32)
    z2 = z2.astype(np.float32)
    w2 = w2.astype(np.float32)

    result = np.zeros_like(d_rbins_squared)[:-1]
    result = result.astype(np.float32)
    results_test = np.zeros_like(result).astype(np.float64)
    count_weighted_pairs_3d_cpu(
        x1, y1, z1, w1, x2, y2, z2, w2, d_rbins_squared.astype(np.float32), results_test)

base_gpairs.run("Gpairs Numba",run_gpairs) 
