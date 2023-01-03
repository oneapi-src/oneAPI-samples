//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <sycl/sycl.hpp>
#include <iomanip>

using namespace sycl;

int main() {
    
    size_t N = 16;
    std::cout << "MATRIX_SIZE    : " << N << "x" << N << std::endl;

    //# Define vectors for matrices
    std::vector<float> matrix_a(N*N);
    std::vector<float> matrix_b(N*N);
    std::vector<float> matrix_c(N*N);
    std::vector<float> matrix_d(N*N);
    
    //# Initialize matrices with values
    float v1 = 2.f;
    float v2 = 3.f;
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++){
            matrix_a[i*N+j] = v1++;
            matrix_b[i*N+j] = v2++;
            matrix_c[i*N+j] = 0.f;
            matrix_d[i*N+j] = 0.f;
    }
    
    //# Define queue with default device for offloading computation
    queue q;
    std::cout << "Offload Device : " << q.get_device().get_info<info::device::name>() << std::endl;
    
    //# Create buffers for matrices
    buffer a(matrix_a);
    buffer b(matrix_b);
    buffer c(matrix_c);

    //# Submit command groups to execute on device
    q.submit([&](handler &h){
        //# Create accessors to copy buffers to the device
        accessor A(a, h, read_only);
        accessor B(b, h, read_only);
        accessor C(c, h, write_only);

        //# Define size for ND-range and work-group size
        range<2> global_size(N,N);
        range<2> work_group_size(N,N);

        //# Create local accessors
        local_accessor<float, 2> A_local(range<2>(N, N), h);
        local_accessor<float, 2> B_local(range<2>(N, N), h);

        //# Parallel Compute Matrix Multiplication
        h.parallel_for(nd_range<2>{global_size, work_group_size}, [=](nd_item<2> item){
            const int i = item.get_global_id(0);
            const int j = item.get_global_id(1);
            const int x = item.get_local_id(0);
            const int y = item.get_local_id(1);

            //# copy from global to local memory
            A_local[x][y] = A[i * N + j];
            B_local[x][y] = B[i * N + j];

            //# barrier to sychronize local memory copy across all work items
            group_barrier(item.get_group());

            //# matrix multiplication computation from local memory
            float temp = 0.f;
            for (int k = 0; k < N; k++) {
                temp += A_local[x][k] * B_local[k][y];
            }
            C[i*N+j] = temp;
        });
    });
    host_accessor ha(c, read_only);
    
    //# Print Output and Verification
    auto FAIL = 0;
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            for(int k=0; k<N; k++){
                matrix_d[i*N+j] += matrix_a[i*N+k] * matrix_b[k*N+j];
            }
            if(matrix_d[i*N+j] != matrix_c[i*N+j]) FAIL = 1;
            std::cout << std::setw(6) << matrix_c[i*N+j] << " ";
        }
        std::cout << "\n";
    }
    if(FAIL == 1) std::cout << "FAIL\n"; else std::cout << "PASS\n";

    return 0;
}

