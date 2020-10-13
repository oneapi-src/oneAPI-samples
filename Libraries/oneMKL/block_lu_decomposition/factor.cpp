//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*      Example of LU factorization of general block tridiagonal matrix
************************************************************************
* Purpose:
* ========  
* Testing LU factorization of block tridiagonal matrix 
*          (D_1  C_1                          )
*          (B_1  D_2  C_2                     )
*          (     B_2  D_3  C_3                )
*          (           .........              )
*          (              B_N-2 D_N-1  C_N-1  )
*          (                    B_N-1  D_N    )
* provided by function dgeblttrf by calculating Frobenius norm of the 
* residual ||A-L*U||. Computation of the residual and its Frobenius norm  
* is done by function resid1 (for source see file auxi.cpp). 
* Input block tridiagonal matrix A is randomly generated.
*/

#include <CL/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <vector>
#include "mkl.h"
#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

using namespace oneapi;

int64_t dgeblttrf(sycl::queue queue, int64_t n, int64_t nb, double* d, double* dl, double* du1, double* du2, int64_t* ipiv);
double resid1( int64_t n, int64_t nb, double* dl,  double* d,  double* du1,  double* du2,  int64_t* ipiv,  double* dlcpy, double* dcpy, double* du1cpy);

template<typename T>
using allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

int main(){

    if (sizeof(MKL_INT) != sizeof(int64_t)) {
        std::cerr << "MKL_INT not 64bit" << std::endl;
        return -1;
    }
    
    int64_t n = 200;
    int64_t nb = 20;

    int64_t info = 0;

    // Asynchronous error handler
    auto error_handler = [&] (sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during asynchronous call
                info = e.info();
                std::cout << "Unexpected exception caught during asynchronous LAPACK operation:\ninfo: " << e.info() << std::endl;
            } catch(sycl::exception const& e) {
                // Handle not LAPACK related exceptions happened during asynchronous call
                std::cout << "Unexpected exception caught during asynchronous operation:\n" << e.what() << std::endl;
                info = -1;
            }
        }
    };

    sycl::device device{sycl::default_selector{}};
    sycl::queue queue(device, error_handler);
    sycl::context context = queue.get_context();

    if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
        std::cerr << "The sample uses double precision, which is not supported" << std::endl;
        std::cerr << "by the selected device. Quitting." << std::endl;
        return 0;
    }

    allocator_t<double> allocator_d(context, device);
    allocator_t<int64_t> allocator_i(context, device);

    std::vector<double, allocator_t<double>> d(nb* n*nb, allocator_d);
    std::vector<double, allocator_t<double>> dl(nb* (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>> du1(nb* (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>> du2(nb* (n-2)*nb, allocator_d);

    std::vector<double, allocator_t<double>> dcpy(nb* n*nb, allocator_d);
    std::vector<double, allocator_t<double>> dlcpy(nb* (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>> du1cpy(nb* (n-1)*nb, allocator_d);

    std::vector<int64_t, allocator_t<int64_t>> ipiv(nb* n, allocator_i);
    std::vector<MKL_INT> iseed = {9, 41, 11, 3};


    std::cout << "Testing accuracy of LU factorization with pivoting" << std::endl;
    std::cout << "of randomly generated block tridiagonal matrix " << std::endl;
    std::cout << "by calculating norm of the residual matrix." << std::endl;

    // Initializing arrays randomly
    LAPACKE_dlarnv(2, iseed.data(), n*nb*nb, d.data());
    LAPACKE_dlarnv(2, iseed.data(), (n-1)*nb*nb, dl.data());
    LAPACKE_dlarnv(2, iseed.data(), (n-1)*nb*nb, du1.data());

    // Copying arrays for testing purposes
    cblas_dcopy(n*nb*nb, d.data(), 1, dcpy.data(), 1);
    cblas_dcopy((n-1)*nb*nb, dl.data(), 1, dlcpy.data(), 1);
    cblas_dcopy((n-1)*nb*nb, du1.data(), 1, du1cpy.data(), 1);

    // Factoring the matrix
    try {
        info = dgeblttrf(queue, n, nb, d.data(), dl.data(), du1.data(), du2.data(), ipiv.data());
    } catch(sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to SYCL API:\n" << e.what() << std::endl;
        info = -1;
    }
    // Check the exit INFO for success
    if(info){
        std::cout << "DGEBLTTRF returned nonzero INFOi = " << info << std::endl;
        return 1;
    }

    // Computing the ratio ||A - LU||_F/||A||_F
    double eps = resid1(n, nb, dl.data(), d.data(), du1.data(), du2.data(), ipiv.data(), dlcpy.data(), dcpy.data(), du1cpy.data());
    std::cout << "||A - LU||_F/||A||_F = " << eps << std::endl;

    return 0;
}
