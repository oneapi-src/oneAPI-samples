
# Copyright 2020 - 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import dpnp as np

from numba_dpex import dpjit


@dpjit
def f1(a, b):
    c = a + b
    return c


def main():
    global_size = 64
    local_size = 32
    N = global_size * local_size
    print("N", N)

    a = np.ones(N, dtype=np.float32)
    b = np.ones(N, dtype=np.float32)

    print(a)
    print(b)

    c = f1(a, b)

    print("RESULT c:", c)
    for i in range(N):
        if c[i] != 2.0:
            print("First index not equal to 2.0 was", i)
            break

    print("Done...")


if __name__ == "__main__":
    main()
