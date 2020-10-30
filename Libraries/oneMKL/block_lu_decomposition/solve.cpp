//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*      Example of solving a system of linear equations with general 
*      block tridiagonal coefficient matrix 
************************************************************************
* Purpose:
* ========  
* Testing solution of a linear system of equations with general block 
* tridiagonal matrix of coefficients
*          D(1)*X(1) +     C(1)*X(2)               = F(1)
*          B(1)*X(1) +     D(2)*X(2) +   C(2)*X(3) = F(2) 
*          B(2)*X(2) +     D(3)*X(3) +   C(3)*X(4) = F(3) 
*      ...
*      B(N-2)*X(N-2) + D(N-1)*X(N-1) + C(N-1)*X(N) = F(N-1) 
*                      B(N-1)*X(N-1) +   D(N)*X(N) = F(N) 
* Here D(J),B(J),C(J) are NB by NB matrices - block matrix coefficients
*     X(J),F(J) are NB by NRHS-matrices - unknowns and RHS components
*
* Solving is done via LU factorization of the coefficient matrix 
* (call DGEBLTTRF) followed by call DGEBLTTRS to solve a system of
* equations with coefficient matrix factored by DGEBLTTRF.
*
* Coefficients and right hand sides are randomly generated.
*
* Testing is done via calculating 
*      max{||F(1)-D(1)*X(1)-C(1)*X(2)||,
*          ||F(2)-B(1)*X(1)-D(1)*X(1)-C(1)*X(2)||,
*           ...
*           ||F(N)-B(N-1)*X(N-1)-D(N)*X(N)||}
*
* ||.|| denotes Frobenius norm of a respective matrix           
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

int64_t dgeblttrf(sycl::queue, int64_t n, int64_t nb, double* d, double* dl, double* du1, double* du2, int64_t* ipiv);
int64_t dgeblttrs(sycl::queue, int64_t n, int64_t nb, int64_t nrhs, double* d, double* dl, double* du1, double* du2, int64_t* ipiv, double* f, int64_t ldf);
double resid2(int64_t n, int64_t nb, int64_t nrhs, double* dl, double* d, double* du1, double* x, int64_t ldx, double* b, int64_t ldb);

template<typename T>
using allocator_t = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

int main() {
    if (sizeof(MKL_INT) != sizeof(int64_t)) {
        std::cerr << "MKL_INT not 64bit" << std::endl;
        return -1;
    }

    int64_t n = 200;
    int64_t nb = 20;
    int64_t nrhs = 10;
    int64_t ldf = nb*n;

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

    std::vector<double, allocator_t<double>>      d(nb * n*nb,     allocator_d);
    std::vector<double, allocator_t<double>>     dl(nb * (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>>    du1(nb * (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>>    du2(nb * (n-2)*nb, allocator_d);
    std::vector<double, allocator_t<double>>     f(ldf * nrhs,     allocator_d);
    std::vector<double, allocator_t<double>>   dcpy(nb * n*nb,     allocator_d);
    std::vector<double, allocator_t<double>>  dlcpy(nb * (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>> du1cpy(nb * (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>>  fcpy(ldf * nrhs,     allocator_d);

    std::vector<int64_t, allocator_t<int64_t>> ipiv(nb * n, allocator_i);
    std::vector<int64_t> iseed = {1, 4, 23, 77};

    // Initializing arrays randomly
    LAPACKE_dlarnv(2, reinterpret_cast<MKL_INT*>(iseed.data()), n*nb*nb, d.data());
    LAPACKE_dlarnv(2, reinterpret_cast<MKL_INT*>(iseed.data()), (n-1)*nb*nb, dl.data());
    LAPACKE_dlarnv(2, reinterpret_cast<MKL_INT*>(iseed.data()), (n-1)*nb*nb, du1.data());
    LAPACKE_dlarnv(2, reinterpret_cast<MKL_INT*>(iseed.data()), n*nb*nrhs, f.data());

    // Copying arrays for testing purposes
    cblas_dcopy(n*nb*nb, d.data(), 1, dcpy.data(), 1);
    cblas_dcopy((n-1)*nb*nb, dl.data(), 1, dlcpy.data(), 1);
    cblas_dcopy((n-1)*nb*nb, du1.data(), 1, du1cpy.data(), 1);
    cblas_dcopy(n*nb*nrhs, f.data(), 1, fcpy.data(), 1);

    std::cout << "Testing accuracy of solution of linear equations system" << std::endl;
    std::cout << "with randomly generated block tridiagonal coefficient" << std::endl;
    std::cout << "matrix by calculating ratios of residuals" << std::endl;
    std::cout << "to RHS vectors norms." << std::endl;

    // LU factorization of the coefficient matrix      
    info = dgeblttrf(queue, n, nb, d.data(), dl.data(), du1.data(), du2.data(), ipiv.data());
    if (info) { 
        std::cout << "DGEBLTTRF returned nonzero INFO = " << info << std::endl;
        return 1;
    }

    // Solving the system of equations using factorized coefficient matrix
    info = dgeblttrs(queue, n, nb, nrhs, d.data(), dl.data(), du1.data(), du2.data(), ipiv.data(), f.data(), ldf);
    if (info) {
        std::cout << -info << "-th parameter in call of dgeblttrs has illegal value" << std::endl;
        return 1;
    } else {
        // computing the residual      
        double eps = resid2(n, nb, nrhs, dlcpy.data(), dcpy.data(), du1cpy.data(), f.data(), ldf, fcpy.data(), ldf);
        std::cout << "max_(i=1,...,nrhs){||ax(i)-f(i)||/||f(i)||} = " << eps << std::endl;
    } 

    return 0;
}
