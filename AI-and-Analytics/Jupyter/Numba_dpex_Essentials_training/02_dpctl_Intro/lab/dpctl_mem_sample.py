# SPDX-FileCopyrightText: 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from math import sqrt
from string import Template
from time import time

import dpctl
import dpctl.memory as dpctl_mem
import dpnp as np

import numba_dpex as ndpx

parser = argparse.ArgumentParser(
    description="Program to compute pairwise distance"
)

parser.add_argument("-n", type=int, default=10, help="Number of points")
parser.add_argument("-d", type=int, default=3, help="Dimensions")
parser.add_argument("-r", type=int, default=1, help="repeat")
parser.add_argument("-l", type=int, default=1, help="local_work_size")

args = parser.parse_args()

# Global work size is equal to the number of points
global_size = ndpx.Range(args.n)
# Local Work size is optional
local_size = ndpx.Range(args.l)

X = np.random.random((args.n, args.d)).astype(np.single)
D = np.empty((args.n, args.n), dtype=np.single)


@ndpx.kernel
def pairwise_distance(X, D, xshape0, xshape1):
    """
    An Euclidean pairwise distance computation implemented as
    a ``kernel`` function.
    """
    idx = ndpx.get_global_id(0)

    d0 = X[idx, 0] - X[idx, 0]
    # for i in range(xshape0):
    for j in range(X.shape[0]):
        d = d0
        for k in range(X.shape[1]):
            tmp = X[idx, k] - X[j, k]
            d += tmp * tmp
        D[idx, j] = sqrt(d)


def driver():
    # measure running time
    times = list()
    for repeat in range(args.r):
        start = time()
        pairwise_distance[ndpx.NdRange(global_size, local_size)](
            X, D, X.shape[0], X.shape[1]
        )
        end = time()

        total_time = end - start
        times.append(total_time)

    return times


def main():
    times = None

    times = driver()

    times = np.asarray(times, dtype=np.float32)
    t = Template("Average time of $runs is = ${timing}")
    tstr = t.substitute(runs=args.r, timing=times.mean())
    print(tstr)

    print("Done...")


if __name__ == "__main__":
    main()
