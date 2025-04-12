//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

//# The following project performs matrix multiplication using oneMKL / DPC++ with Unified Shared Memory (USM)
//# We will execute the simple operation A * B = C
//# The matrix B is set equal to the identity matrix such that A * B = A * I
//# After performing the computation, we will verify A * I = C -> A = C

namespace mkl = oneapi::mkl;  //# shorten mkl namespace

int main() {

    //# dimensions
    int m = 3, n = 3, k = 3;
    //# leading dimensions
    int ldA = 3, ldB = 3, ldC = 3;
    //# scalar multipliers
    double alpha = 1.0, beta = 1.0;
    //# transpose status of matrices
    mkl::transpose transA = mkl::transpose::nontrans;
    mkl::transpose transB = mkl::transpose::nontrans;

    //### Step 1 - Observe the definition of an asynchronous exception handler.
    //# This function object will later be supplied to the queue.
    //# It is designed to handle errors thrown while device code executes.
    auto async_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
        }
    };

    //### Step 2 - Create a device object.
    //# Device selectors are used to specify the type of a device.
    //# Uncomment _one_ of the following three lines to select a device.
    // sycl::device device = sycl::device(sycl::default_selector());  //# default_selector returns a device based on a performance heuristic
    // sycl::device device = sycl::device(sycl::cpu_selector());      //# cpu_selector returns a cpu device
    // sycl::device device = sycl::device(sycl::gpu_selector());      //# gpu_selector returns a gpu device
    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";

    //### Step 3 - Create a queue object.
    //# A queue accepts a single device, and optionally, an exception handler.
    //# Uncomment the following line to initialize a queue with our device and handler.
    // sycl::queue queue(device, async_handler);

    //### Step 4 - Create a sycl event and allocate USM
    //# The later execution of the gemm operation is tied to this event
    //# The gemm operation will also make use of a vector of sycl events we can call 'gemm_dependencies'
    sycl::event gemm_done;
    std::vector<sycl::event> gemm_dependencies;
    //# Here, we allocate USM pointers for each matrix, using the special 'malloc_shared' function
    //# Make sure to template the function with the correct precision, and pass in our queue to the function call
    double *A_usm = sycl::malloc_shared<double>(m * k, queue);
    double *B_usm = sycl::malloc_shared<double>(k * n, queue);
    double *C_usm = sycl::malloc_shared<double>(m * n, queue);

    //# define matrix A as the 3x3 matrix
    //# {{ 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A_usm[i*m+j] = (double)(i*m+j) + 1.0;
        }
    }
    
    //# define matrix B as the identity matrix
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) B_usm[i*k+j] = 1.0;
            else B_usm[i*k+j] = 0.0;
        }
    }
    
    //# initialize C as a 0 matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C_usm[i*m+j] = 0.0;
        }
    }

    //### Step 5 - Execute gemm operation.
    //# Here, we fill in the familiar parameters for the gemm operation.
    //# However, we must also pass in the queue as the first parameter.
    //# We must also pass in our list of dependencies as the final parameter.
    //# We are also passing in our USM pointers as opposed to a buffer or raw data pointer.
    gemm_done = mkl::blas::gemm(queue, transA, transB, m, n, k, alpha, A_usm, ldA, B_usm, ldB, beta, C_usm, ldC, gemm_dependencies);

    //# We must now wait for the given event to finish before accessing any data involved in the operation
    //# Otherwise, we may access data before the operation has completed, or before it has been returned to the host
    gemm_done.wait();

    int status = 0;

    //# verify C matrix using USM data
    std::cout << "\n";
    std::cout << "C = \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A_usm[i*m+j] != C_usm[i*m+j]) status = 1;
            std::cout << C_usm[i*m+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    //# free usm pointers
    sycl::free(A_usm, queue);
    sycl::free(B_usm, queue);
    sycl::free(C_usm, queue);

    status == 0 ? std::cout << "Verified: A = C\n" : std::cout << "Failed: A != C\n";
    return status;
}