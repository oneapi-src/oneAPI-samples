//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Code for STREAM:
#include <iostream>
#include <omp.h>
#include <cstdint>

// compile via:
// icpx -O2 -fiopenmp -fopenmp-targets=spir64 ./stream.cpp

int main()
{
    constexpr int64_t N = 256 * 1e6;
    constexpr int64_t bytes = N * sizeof(int64_t);

    int64_t *a = static_cast<int64_t *>(malloc(bytes));
    int64_t *b = static_cast<int64_t *>(malloc(bytes));
    int64_t *c = static_cast<int64_t *>(malloc(bytes));

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
        if (irep == 10)
            time = omp_get_wtime();

        #pragma omp target teams distribute parallel for simd
        for (int i = 0; i < N; ++i)
        {
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
            std::cout << "wrong results!" << std::endl;
            exit(1);
        }
    }

    const int64_t streamed_bytes = 3 * N * sizeof(int64_t);

    std::cout << "bandwidth = " << (streamed_bytes / time) * 1E-9
        << " GB/s" << std::endl;
}
