//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Code for 3D STREAM
#include <iostream>
#include <omp.h>
#include <cassert>

// compile via:
// icpx -O2 -fiopenmp -fopenmp-targets=spir64 ./stream_3D.cpp

int main(int argc, char **argv)
{
    const int device_id = omp_get_default_device();
    const int desired_total_size = 32 * 512 * 16384;
    const std::size_t bytes = desired_total_size * sizeof(int64_t);

    std::cout << "memory footprint = " << 3 * bytes * 1E-9 << " GB"
              << std::endl;

    int64_t *a = static_cast<int64_t*>(omp_target_alloc_device(bytes, device_id));
    int64_t *b = static_cast<int64_t*>(omp_target_alloc_device(bytes, device_id));
    int64_t *c = static_cast<int64_t*>(omp_target_alloc_device(bytes, device_id));

    const int min = 64;
    const int max = 32768;

    for (int lx = min; lx < max; lx *= 2)
    {
        for (int ly = min; ly < max; ly *= 2)
        {
            for (int lz = min; lz < max; lz *= 2)
            {
                const int total_size = lx * ly * lz;
                if (total_size != desired_total_size)
                    continue;

                std::cout << "lx=" << lx << " ly=" << ly << " lz="
                    << lz << ", ";

                #pragma omp target teams distribute parallel for simd
                for (int i = 0; i < total_size; ++i)
                {
                    a[i] = i + 1;
                    b[i] = i - 1;
                    c[i] = 0;
                }

                const int no_max_rep = 40;
                const int warmup = 10;
                double time;
                for (int irep = 0; irep < no_max_rep + warmup; ++irep)
                {
                    if (irep == warmup) time = omp_get_wtime();

                    #pragma omp target teams distribute parallel for simd collapse(3)
                    for (int iz = 0; iz < lz; ++iz)
                    {
                        for (int iy = 0; iy < ly; ++iy)
                        {
                            for (int ix = 0; ix < lx; ++ix)
                            {
                                const int index = ix + iy * lx + iz * lx * ly;
                                c[index] = a[index] + b[index];
                            }
                        }
                    }
                }
                time = omp_get_wtime() - time;
                time = time / no_max_rep;

                const int64_t streamed_bytes = 3 * total_size * sizeof(int64_t);

                std::cout << "bandwidth = " << (streamed_bytes / time) * 1E-9
                          << " GB/s" << std::endl;

                #pragma omp target teams distribute parallel for simd
                for (int i = 0; i < total_size; ++i)
                {
                    assert(c[i] == 2 * i);
                }
            }
        }
    }

    omp_target_free(a, device_id);
    omp_target_free(b, device_id);
    omp_target_free(c, device_id);
}
