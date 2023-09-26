
# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import base_pair_wise_gpu_graph
import numba_dpex as nbdx
import dpnp as np


@nbdx.kernel
def pairwise_python(X1, X2, D):
    i = nbdx.get_global_id(0)

    N = X2.shape[0]
    O = X1.shape[1]
    for j in range(N):
        d = 0.0
        for k in range(O):
            tmp = X1[i, k] - X2[j, k]
            d += tmp * tmp
        D[i, j] = np.sqrt(d)


def pw_distance(X1, X2, D):
    pairwise_python[X1.shape[0],](X1, X2, D)

base_pair_wise_gpu_graph.run("Pairwise Distance Kernel", pw_distance)
