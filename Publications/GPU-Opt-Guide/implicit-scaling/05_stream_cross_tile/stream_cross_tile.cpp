//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Code for cross tile stream
#include <iostream>
#include <omp.h>

// compile via:
// icpx -O2 -fiopenmp -fopenmp-targets=spir64 ./stream_cross_tile.cpp
// run via:
// EnableWalkerPartition=1 ZE_AFFINITY_MASK=0 ./a.out

int main()
{
    constexpr int64_t N = 256 * 1e6;
    constexpr int64_t bytes = N * sizeof(int);

    // vary n_th from 1 to 8 to change cross-tile traffice
    constexpr int n_th = 4;

    std::cout << "array size = " << bytes * 1e-9 << " GB" << std::endl;

    int *a = static_cast<int *>(malloc(bytes));
    int *b = static_cast<int *>(malloc(bytes));
    int *c = static_cast<int *>(malloc(bytes));

    #pragma omp target enter data map(alloc:a[0:N])
    #pragma omp target enter data map(alloc:b[0:N])
    #pragma omp target enter data map(alloc:c[0:N])

    for (int i = 0; i < N; ++i)
    {
        a[i] = i + 1;
        b[i] = i - 1;
    }

    #pragma omp target update to(a[0:N])
    #pragma omp target update to(b[0:N])

    const int no_max_rep = 100;
    double time;

    for (int irep = 0; irep < no_max_rep + 10; ++irep)
    {
        if (irep == 10) time = omp_get_wtime();

        #pragma omp target teams distribute parallel for \
            simd simdlen(32) thread_limit(256)
        for (int j = 0; j < N; ++j)
        {
            const int cache_line_id = j / 16;
            int i;
            if ((cache_line_id % n_th) == 0)
            {

                i = (j + N / 2) % N;
            }
            else
            {

                i = j;
            }

            c[i] = a[i] + b[i];
        }
    }
    time = omp_get_wtime() - time;
    time = time / no_max_rep;

    #pragma omp target update from(c[0:N])

    for (int i = 0; i < N; ++i)
    {
        if (c[i] != 2 * i)
        {
            std::cout << "wrong results at i " << i << std::endl;
            exit(1);
        }
    }

    const int64_t streamed_bytes = 3 * N * sizeof(int);

    std::cout << "bandwidth = " << (streamed_bytes / time) * 1E-9 << " GB/s"
        << std::endl;
    std::cout << "cross-tile traffic = " << (1 / (double)n_th) * 100 << "%"
        << std::endl;
}
