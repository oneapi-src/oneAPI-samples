//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Code for cross stack stream
#include <iostream>
#include <omp.h>

// compile via:
// icpx -O2 -fiopenmp -fopenmp-targets=spir64 ./stream_cross_stack.cpp
// run via:
// EnableWalkerPartition=1 ZE_AFFINITY_MASK=0 ./a.out

template <int cross_stack_fraction>
void cross_stack_stream() {

    constexpr int64_t size = 256*1e6;
    constexpr int64_t bytes = size * sizeof(int64_t);

    int64_t *a = static_cast<int64_t*>(malloc( bytes ));
    int64_t *b = static_cast<int64_t*>(malloc( bytes ));
    int64_t *c = static_cast<int64_t*>(malloc( bytes ));
    #pragma omp target enter data map( alloc:a[0:size] )
    #pragma omp target enter data map( alloc:b[0:size] )
    #pragma omp target enter data map( alloc:c[0:size] )

    for ( int i = 0; i < size; ++i ) {

        a[i] = i + 1;
        b[i] = i - 1;
        c[i] = 0;
    }

    #pragma omp target update to( a[0:size] )
    #pragma omp target update to( b[0:size] )
    #pragma omp target update to( c[0:size] )

    const int num_max_rep = 100;

    double time;

    for ( int irep = 0; irep < num_max_rep+10; ++irep ) {

        if ( irep == 10 ) time = omp_get_wtime();

        #pragma omp target teams distribute parallel for simd
        for ( int j = 0; j < size; ++j ) {

            const int cache_line_id = j / 16;

            int i;

            if ( (cache_line_id%cross_stack_fraction) == 0 ) {

                i = (j+size/2)%size;
            }
            else {

                i = j;
            }

            c[i] = a[i] + b[i];
        }
    }
    time = omp_get_wtime() - time;
    time = time/num_max_rep;

    #pragma omp target update from( c[0:size] )

    for ( int i = 0; i < size; ++i ) {

        if ( c[i] != 2*i ) {

            std::cout << "wrong results!" << std::endl;
            exit(1);
        }
    }

    const int64_t streamed_bytes = 3 * size * sizeof(int64_t);

    std::cout << "cross_stack_percent = " << (1/(double)cross_stack_fraction)*100
              << "%, bandwidth = " << (streamed_bytes/time) * 1E-9 << " GB/s" << std::endl;
}

int main() {

    cross_stack_stream< 1>();
    cross_stack_stream< 2>();
    cross_stack_stream< 4>();
    cross_stack_stream< 8>();
    cross_stack_stream<16>();
    cross_stack_stream<32>();
}
