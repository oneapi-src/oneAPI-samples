//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include <vector>

//# sycl namespace
#include <sycl/sycl.hpp>          
using namespace sycl;

//# oneMKL DPC++ interface for BLAS functions

#include "oneapi/mkl/blas.hpp"  
// # shorten mkl namespace
namespace mkl = oneapi::mkl;    

//# The following project performs matrix multiplication using oneMKL / DPC++ with buffers.
//# We will execute the simple operation A * B = C
//# The matrix B is set equal to the identity matrix such that A * B = A * I
//# After performing the computation, we will verify A * I = C -> A = C



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

    //### Step 2 - Create a device object. (create device and q in one step)
    //# Device selectors are used to specify the type of a device.
    //# Uncomment _one_ of the following three lines to select a device.
    queue q(default_selector_v, async_handler);  //# default_selector returns a device based on a performance heuristic
    // queue q(cpu_selector_v);      //# cpu_selector returns a cpu device
    // queue q(gpu_selector_v);     //# gpu_selector returns a gpu device
    // queue q;
    //# Print actual device used
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    //### Step 4 - Create buffers to hold our matrix data.
    //# Buffer objects can be constructed given a container
    //# Observe the creation of buffers for matrices A and B.
    //# Try and create a third buffer for matrix C called C_buffer.
    //# The solution is shown in the hidden cell below.
    buffer A_buffer(A);
    buffer B_buffer(B);
    /* define C_buffer below */
    buffer C_buffer(C);
    

    //### Step 5 - Execute gemm operation.
    //# Here, we need only pass in our queue and other familiar matrix multiplication parameters.
    //# This includes the dimensions and data buffers for matrices A, B, and C.
    mkl::blas::gemm(q, transA, transB, m, n, k, alpha, A_buffer, ldA, B_buffer, ldB, beta, C_buffer, ldC);

    //# we cannot explicitly transfer memory to/from the device when using buffers
    //# that is why we must use this operation to ensure result data is returned to the host
    q.wait_and_throw();  //# block until operation completes, throw any errors

    //### Step 6 - Observe creation of accessors to retrieve data from A_buffer and C_buffer.
    accessor A_acc(A_buffer,read_only);
    accessor C_acc(C_buffer,read_only);

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
