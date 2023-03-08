##==============================================================
## Copyright © Intel Corporation
##
## SPDX-License-Identifier: Apache-2.0
## =============================================================

import dpnp as np
import numba_dpex as ndpex
import timeit


@ndpex.kernel
def atomic_reduction(a):
    idx = ndpex.get_global_id(0)
    ndpex.atomic.add(a, 0, a[idx])


def main():
    N = 1024
    a = np.arange(N)   

    #print("Using device ...")
    #print(a.device)

    atomic_reduction[N, ndpex.DEFAULT_LOCAL_SIZE](a)
    #print("Reduction sum =", a[0])

    #print("Done...")


if __name__ == "__main__":
    t = timeit.Timer(lambda: main())    
    print("Time to calculate reduction using atomics",t.timeit(500),"seconds")
