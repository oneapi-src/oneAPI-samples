//==============================================================
// Matrix Multiplication: SYCL Local Accessor
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <CL/sycl.hpp>

using namespace sycl;

void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N, size_t M) {
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << " | WORK_GROUP_SIZE= " << M << "x" << M << "\n";

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

        //# Define size for ND-range and work-group size
        range<2> global_size(N,N);
        range<2> work_group_size(M,M);

        //# Create local accessors
        accessor<float, 2, access::mode::read_write, access::target::local> A_tile(range<2>(M, M), h);
        accessor<float, 2, access::mode::read_write, access::target::local> B_tile(range<2>(M, M), h);

        //# Parallel Compute Matrix Multiplication
        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            const int x = item.get_local_id(0);
            const int y = item.get_local_id(1);

            float temp = 0.f;
            int k;
            for (int t = 0; t < N; t+=M) {
                A_tile[x][y] = A[i * N + (t + y)];
                B_tile[x][y] = B[(t + x) * N + j];
                item.barrier(access::fence_space::local_space);
                for (k = 0; k < M; k++) {
                    temp += A_tile[x][k] * B_tile[k][y];
                }
                item.barrier(access::fence_space::local_space);
            }
            C[i*N+j] = temp;
        });
    });
    c.get_access<access::mode::read>();
    
    //# print kernel compute duration from event profiling
    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds\n";
}
