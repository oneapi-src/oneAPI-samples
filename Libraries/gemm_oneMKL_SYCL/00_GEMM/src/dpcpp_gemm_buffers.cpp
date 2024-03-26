//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>          //# sycl namespace
#include "oneapi/mkl/blas.hpp"  //# oneMKL DPC++ interface for BLAS functions

//# The following project performs matrix multiplication using oneMKL / DPC++ with buffers.
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
    //# matrix data
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    std::vector<double> B = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> C = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

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

    //### Step 4 - Create buffers to hold our matrix data.
    //# Buffer objects can be constructed given a container
    //# Observe the creation of buffers for matrices A and B.
    //# Try and create a third buffer for matrix C called C_buffer.
    //# The solution is shown in the hidden cell below.
    sycl::buffer A_buffer(A);
    sycl::buffer B_buffer(B);
    /* define C_buffer here */

    //### Step 5 - Execute gemm operation.
    //# Here, we need only pass in our queue and other familiar matrix multiplication parameters.
    //# This includes the dimensions and data buffers for matrices A, B, and C.
    mkl::blas::gemm(queue, transA, transB, m, n, k, alpha, A_buffer, ldA, B_buffer, ldB, beta, C_buffer, ldC);

    //# we cannot explicitly transfer memory to/from the device when using buffers
    //# that is why we must use this operation to ensure result data is returned to the host
    queue.wait_and_throw();  //# block until operation completes, throw any errors

    //### Step 6 - Observe creation of accessors to retrieve data from A_buffer and C_buffer.
    sycl::host_accessor A_acc(A_buffer, sycl::read_only);
    sycl::host_accessor C_acc(C_buffer, sycl::read_only);

    int status = 0;

    // verify C matrix using accessor to observe values held in C_buffer
    std::cout << "\n";
    std::cout << "C = \n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A_acc[i*m+j] != C_acc[i*m+j]) status = 1;
            std::cout << C_acc[i*m+j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    status == 0 ? std::cout << "Verified: A = C\n" : std::cout << "Failed: A != C\n";
    return status;
}