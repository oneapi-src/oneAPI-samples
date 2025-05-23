//==============================================================
// Matrix Multiplication: SYCL Matrix Multiplication Common WG
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================


#include <sycl/sycl.hpp>
#include <getopt.h>
#include <ctime>
#include <chrono>
#include <cmath>

using namespace sycl;

//# matrix multiplication kernel implementation in mm_dpcpp_*.cpp
void mm_kernel(queue &q, std::vector<float> &matrix_a, std::vector<float> &matrix_b, std::vector<float> &matrix_c, size_t N, size_t M);

//# floating point error verification function
bool almost_equal(float a, float b){
    float tolerance = 1e-6;
    float diff = std::fabs(a - b);
    a = std::fabs(a);
    b = std::fabs(b);
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
                std::cout << "Usage   : ./a.out -n <MATRIX_SIZE> -m <WORK_GROUP_SIZE> -v -p\n\n";
                std::cout << "          [-n] size for matrix, eg: 1024\n";
                std::cout << "          [-m] size of work_group, eg: 8/16\n";
                std::cout << "          [-v] verify output with linear computation on cpu\n";
                std::cout << "          [-p] print output matrix\n";
                std::cout << "Example : ./a.out -n 1024 -m 16 -v -p\n\n";
                std::exit(0);
        }

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
    queue q(property::queue::enable_profiling{});
    std::cout << "Offload Device        : " << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "max_work_group_size   : " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";
    
    std::cout << "matrix_size           : " << N << "x" << N << "\n";
    
    // find valid work-group sizes to try for performance.
    std::vector<int> work_group_sizes;
    auto max_work_group_size = q.get_device().get_info<info::device::max_work_group_size>();
    int work_group_dim_size = std::sqrt(max_work_group_size);
    work_group_dim_size = work_group_dim_size - work_group_dim_size % 2; 
    while (work_group_dim_size >= 2){
        if (N % work_group_dim_size == 0) work_group_sizes.push_back(work_group_dim_size);
        work_group_dim_size =  work_group_dim_size - 2;
    }
    std::cout << "valid_wg_sizes        : " ;
    for(int i=0;i<work_group_sizes.size();i++) std::cout << work_group_sizes[i] << "x" << work_group_sizes[i] << " ";
    std::cout << "\n";
    
    // find optimal work-group size for the offload device
    int optimal_work_group_dim_size = 0;
    for(int i=0;i<work_group_sizes.size();i++){
        if(work_group_sizes[i] % 8 == 0) {optimal_work_group_dim_size = work_group_sizes[i]; break;}
    }
    for(int i=0;i<work_group_sizes.size();i++){
        if(work_group_sizes[i] % 16 == 0) {optimal_work_group_dim_size = work_group_sizes[i]; break;}
    }
    for(int i=0;i<work_group_sizes.size();i++){
        if(work_group_sizes[i] % 32 == 0) {optimal_work_group_dim_size = work_group_sizes[i]; break;}
    }
    std::cout << "optimal_wg_size       : " << optimal_work_group_dim_size << "x" << optimal_work_group_dim_size << "\n";
    if(M ==0) M = optimal_work_group_dim_size;
    
    //# get start time
    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    //# Call matrix multiplication kernel implementation
    mm_kernel(q, matrix_a, matrix_b, matrix_c, N, M);
    
    //# print kernel compute duration from host
    auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
    std::cout << "Compute Duration      : " << duration / 1e+9 << " seconds\n";
    
    //# Print Output if -p in cmd-line
    if (PRINT_OUTPUT_MATRIX){
        for (int i=0; i<N; i++){
            for (int j=0; j<N; j++){
                std::cout << matrix_c[i*N+j] << " ";
            }
            std::cout << "\n";
        }
    } else {
        std::cout << " [0][0] = " << matrix_c[0] << "\n";
    }
    
    //# Compute local and compare with offload computation if -v in cmd-line
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
            std::cout << "FAIL\n";
        } else {
            std::cout << "PASS\n";
        }
    }
    return 0;
}




