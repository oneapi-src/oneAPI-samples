//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Matrix Multiplication is a simple program that multiplies together two
//     large matrices and verifies the results.
// This samples uses the oneAPI Math Kernel Library (oneMKL) to accelerate
//     the computation.

// The test is updated based on oneAPI samples oneAPI-samples/Libraries/oneMKL/matrix_mul_mkl
#include <iostream>
#include <iomanip>
#include <limits>

#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "dpc_common.hpp"

#ifndef USE_DOUBLE
#define FLOAT   float
#else
#define FLOAT   double
#endif

FLOAT rand_uniform();
bool verify_result(int m, int n, int k, int ldc, FLOAT *C, FLOAT *C_reference);

#define WARMUP	 10
#define LOOPS   100
//default matrix size 8192x8192
#define MSIZE   8192
#define VERIFY_RESULT   False


using namespace std ;

int main(int argc, char* argv[])
{
    try {

        int msize = MSIZE;
        int loops = LOOPS;
        int verify = 0;

        // Initialize data for GEMM. The full GEMM operation is:
        //
        //      C = alpha * op(A) * op(B) + beta * C
        //
        // where alpha, beta are scalar values, and op(...) represents
        // optional matrix transposition.
        //
        // For this simple matrix multiplication, no transposition is needed.
        // 
        // By choosing alpha = 1, beta = 0, GEMM will calculate C = A * B.
        //
        // In this example, matrices are stored in row-major layout.

        auto transA = oneapi::mkl::transpose::nontrans;
        auto transB = oneapi::mkl::transpose::nontrans;

        // Matrix data sizes.
        // 
        // A is m x k
        // B is k x n  --> product C is m x n
        int m = msize;
        int k = msize;
        int n = msize;

        cout << "Problem size: "
                  << " A (" << m << 'x' << k << ") *"
                  << " B (" << k << 'x' << n << ")  --> "
                  << " C (" << m << 'x' << n << ")\n";
        
        cout << "Benchmark interations: " << loops << endl;

        // Leading dimensions of data. For row-major matrices, the leading
        // dimension is the stride between adjacent rows.
        int lda = k;
        int ldb = n;
        int ldc = n;

        // Scaling factors.
        FLOAT alpha = 1.0f;
        FLOAT beta = 0.0f;

        // Create a queue on the default device.
        sycl::queue device_queue{sycl::default_selector_v};

        std::cout << "Device: "
                  << device_queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        // Allocate shared memory for matrices.
        const size_t alignment = 4096;
        auto a = sycl::aligned_alloc_host<FLOAT>(alignment, m * k, device_queue);
        auto b = sycl::aligned_alloc_host<FLOAT>(alignment, k * n, device_queue);
        auto c = sycl::aligned_alloc_host<FLOAT>(alignment, m * n, device_queue);

        auto C_reference = (FLOAT *) calloc(m * n, sizeof(FLOAT));

        if (!a || !b || !c || !C_reference) {
            std::cerr << "Could not allocate memory for matrices." << std::endl;
            exit(1);
        }

        // Initialize matrix data.
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                a[i * lda + j] = rand_uniform();

        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                b[i * ldb + j] = rand_uniform();

        auto A = sycl::aligned_alloc_device<FLOAT>(alignment, m * k, device_queue);
        auto B = sycl::aligned_alloc_device<FLOAT>(alignment, m * n, device_queue);
        auto C = sycl::aligned_alloc_device<FLOAT>(alignment, m * n, device_queue);
        device_queue.wait();

        device_queue.memcpy(A, &(a[0]), m * k * sizeof(FLOAT));
        device_queue.memcpy(B, &(b[0]), k * n * sizeof(FLOAT));
        device_queue.memcpy(C, &(c[0]), m * n * sizeof(FLOAT));
        device_queue.wait();
        

        // Call GEMM to do matrix multiplication, asynchronously.
        std::cerr << "Launching oneMKL GEMM calculation..." << std::endl;
        dpc_common::TimeInterval timer;
        double start_time = 0.0;

        //warm up
	for (int w=0; w < WARMUP; w++)
	{
            oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                            alpha, A, lda, B, ldb, beta, C, ldc);
	}
	device_queue.wait_and_throw();

        start_time = timer.Elapsed();
        for (int l=0; l < loops; l++)
        {
            oneapi::mkl::blas::row_major::gemm(device_queue, transA, transB, m, n, k,
                                            alpha, A, lda, B, ldb, beta, C, ldc);
        }
        // Wait for oneMKL computation to complete.        
        device_queue.wait_and_throw();

        double stop_time = timer.Elapsed();
        double avg_gemm_time = (stop_time - start_time)/loops;

        double gflops = 2.0 * (double)m * (double)m * (double)m;
        #ifdef USE_DOUBLE
            cout << "DGEMM performance : " << gflops / avg_gemm_time * 1.e-9 << " GFLOPS" << endl;
        #else
            cout << "SGEMM performance : " << gflops / avg_gemm_time * 1.e-9 << " GFLOPS" << endl;
        #endif

        
        if(verify)
        {
            // While calculation occurs, compute reference result to check accuracy.
            std::cerr << "Performing reference calculation..." << std::endl;
            for (int i = 0; i < m; i++)
                for (int h = 0; h < k; h++)
                    for (int j = 0; j < n; j++)
                        C_reference[i * ldc + j] += a[i * lda + h] * b[h * ldb + j];        
            // Check results for accuracy.
           device_queue.memcpy(&(c[0]), C, m*n*sizeof(FLOAT)).wait();
           verify_result(m, n, k, ldc, c, C_reference);
        }
        

        // Free memory.
        free(A, device_queue);
        free(B, device_queue);
        free(C, device_queue);
        free(C_reference);        
	free(a, device_queue);
	free(b, device_queue);
	free(c, device_queue);

    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);
    }
}

FLOAT rand_uniform()
{
    return static_cast <FLOAT> (rand()) / static_cast <FLOAT> (RAND_MAX);
}

bool verify_result(int m, int n, int k, int ldc, FLOAT *C, FLOAT *C_reference)
{
    FLOAT tolerance = 1e-3;
    bool ok = true;

    // Compare host side results with the result buffer from device side: print
    // fail data 5 times only.
    int printf_count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            auto idx = i * ldc + j;
            auto abs_diff = std::abs(C[idx] - C_reference[idx]);

            if (abs_diff > tolerance && printf_count++ < 5) {
                std::cerr << "The result is incorrect for element "
                          << '[' << i << ", " << j << ']'
                          << ", expected: " << C_reference[idx]
                          << ", but got: " << C[idx] << std::endl;
                ok = false;
            }
        }
    }

    if (ok)
        std::cout << "Results are accurate with tolerance = " << tolerance << endl;
    else
        std::cout << "Results may not be accurate with tolerance = " << tolerance << endl;

    return ok;
}
