# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_l2_distance
import numpy as np
import numba

@numba.jit(nopython=True,parallel=True,fastmath=True)
def l2_distance(a,b):
    sub = a-b
    sq = np.square(sub)
    sum = np.sum(sq)
    d = np.sqrt(sum)
    return d

base_l2_distance.run("l2 distance", l2_distance)
