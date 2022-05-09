//==============================================================
// Matrix Multiplication: SYCL Basic Parallel Kernel
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <CL/sycl.hpp>

using namespace sycl;

void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N, size_t M) {
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << "\n";
    
    //# Create buffers for matrices
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    //# Submit command groups to execute on device
    auto e = q.submit([&](handler &h){
        //# Create accessors to copy buffers to the device
        auto A = a.get_access<access::mode::read>(h);
        auto B = b.get_access<access::mode::read>(h);
        auto C = c.get_access<access::mode::write>(h);

        //# Parallel Compute Matrix Multiplication
        h.parallel_for(range<2>{N,N}, [=](item<2> item){
            const int i = item.get_id(0);
            const int j = item.get_id(1);
            for (int k = 0; k < N; k++) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        });
    });
    c.get_access<access::mode::read>();
    
    //# print kernel compute duration from event profiling
    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
}

