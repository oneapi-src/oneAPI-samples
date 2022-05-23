//==============================================================
// Matrix Multiplication: SYCL Basic Parallel Kernel
//==============================================================
// Copyright © 2021 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <CL/sycl.hpp>
#include <ctime>
#include <chrono>
#include <getopt.h>

using namespace sycl;

//# floating point error verification function
bool almost_equal(float a, float b){
    float tolerance = 1e-6;
    float diff = fabs(a - b);
    a = fabs(a);
    b = fabs(b);
    float bigger = (b > a) ? b : a;
    if(diff <= bigger * tolerance) return true;
    return false;
}

int main(int argc, char *argv[]) {
    
    size_t N = 1024;
    size_t M = 0;
    int VERIFY = 0;
    int PRINT_OUTPUT_MATRIX = 0;
    
    //# command line arguments
    int arg;
    while ((arg = getopt (argc, argv, "n:m:vp")) != -1)
        switch (arg){
            case 'n':
                N = std::atoi(optarg);
                break;
            case 'm':
                M = std::atoi(optarg);
                break;
            case 'v':
                VERIFY = 1;
                break;
            case 'p':
                PRINT_OUTPUT_MATRIX = 1;
                break;
            case 'h':
                std::cout << std::endl;
                std::cout << "Usage   : ./a.out -n <MATRIX_SIZE> -m <WORK_GROUP_SIZE> -v -p" << std::endl << std::endl;
                std::cout << "          [-n] size for matrix, eg: 1024" << std::endl;
                std::cout << "          [-m] size of work_group, eg: 8/16" << std::endl;
                std::cout << "          [-v] verify output with linear computation on cpu" << std::endl;
                std::cout << "          [-p] print output matrix" << std::endl;
                std::cout << "Example : ./a.out -n 1024 -m 16 -v -p" << std::endl << std::endl;
                std::exit(0);
        }
    std::cout << "Configuration         : MATRIX_SIZE= " << N << "x" << N << std::endl;

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
    
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    //# Define queue with default device for offloading computation
    queue q{property::queue::enable_profiling{}};
    event e;
    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << std::endl;
    {
        //# Create buffers for matrices
        buffer a(matrix_a);
        buffer b(matrix_b);
        buffer c(matrix_c);

        //# Submit command groups to execute on device
        e = q.submit([&](handler &h){
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
    }
    
    auto kernel_duration = (e.get_profiling_info<info::event_profiling::command_end>() - e.get_profiling_info<info::event_profiling::command_start>());
    std::cout << "Kernel Execution Time : " << kernel_duration / 1e+9 << " seconds" << std::endl;
    
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    std::cout << "Compute Duration      : " << duration / 1e+9 << " seconds" << std::endl;
    
    //# Print Output
    if (PRINT_OUTPUT_MATRIX){
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                std::cout << matrix_c[i*N+j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << " [0][0] = " << matrix_c[0] << std::endl;
    }
    
    //# Compute local and compare with offload computation
    if (VERIFY){
        int fail = 0;
        for(int i=0; i<N; i++){
            for (int j = 0; j < N; j++) {
                for(int k=0; k<N; k++){
                    matrix_d[i*N+j] += matrix_a[i*N+k] * matrix_b[k*N+j];
                }
                if(!almost_equal(matrix_c[i*N+j], matrix_d[i*N+j])) fail = 1;
            }
        }
        if(fail == 1){
            std::cout << "FAIL" << std::endl;
        } else {
            std::cout << "PASS" << std::endl;
        }
    }
    return 0;
}

