//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*  Content:
*      Example of solving a system of linear equations with symmetric
*      positive definite block tridiagonal coefficient matrix Cholesky
*      factored
************************************************************************
* Purpose:
* ========
* Testing accuracy of solution of a system of linear equations A*X=F
* with a symmetric positive definite block tridiagonal coefficient
* matrix A
*  | D_1  B_1^t                           |
*  | B_1  D_2   B_2^t                     |
*  |      B_2  D_3   B_3^t                |
*  |         .     .      .               |
*  |             .     .      .           |
*  |               B_N-2  D_N-1   B_N-1^t |
*  |                      B_N-1   D_N     |
* preliminarily Cholesky factored as follows:
*      | L_1                  | | L_1^t  C_1^t                   |
*      | C_1  L_2             | |        L_2^t  C_2^t            |
*  A = |    .     .           |*|             .       .          |
*      |        .     .       | |                  .     C_N-1^t |
*      |           C_N-1  L_N | |                        L_N^t   |
*
* To test the solution function TES_RES1 is called.
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

template<typename T>
using allocator_t = cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared>;

int64_t dpbltrf(cl::sycl::queue queue, int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb);
int64_t dpbltrs(cl::sycl::queue queue, int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf);

double test_res1(int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf, double* x, int64_t ldx );

int main() {

    if (sizeof(MKL_INT) != sizeof(int64_t)) {
        std::cerr << "MKL_INT not 64bit" << std::endl;
        return -1;
    }

    int64_t info = 0;

    // Asynchronous error handler
    auto error_handler = [&] (cl::sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during asynchronous call
                info = e.info();
                std::cout << "Unexpected exception caught during asynchronous LAPACK operation:\ninfo: " << e.info() << std::endl;
            } catch(cl::sycl::exception const& e) {
                // Handle not LAPACK related exceptions happened during asynchronous call
                std::cout << "Unexpected exception caught during asynchronous operation:\n" << e.what() << std::endl;
                info = -1;
            }
        }
    };

    cl::sycl::device device{cl::sycl::default_selector{}};
    cl::sycl::queue queue(device, error_handler);
    cl::sycl::context context = queue.get_context();

    if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
        std::cerr << "The sample uses double precision, which is not supported" << std::endl;
        std::cerr << "by the selected device. Quitting." << std::endl;
        return 0;
    }

    allocator_t<double> allocator_d(context, device);

    int64_t n = 200;
    int64_t nb = 20;
    int64_t nrhs = 10;
    int64_t ldf = nb*n;


    std::vector<double, allocator_t<double>> d(nb * n*nb,     allocator_d);
    std::vector<double, allocator_t<double>> b(nb * (n-1)*nb, allocator_d);
    std::vector<double, allocator_t<double>> f(ldf * nrhs,    allocator_d);
    std::vector<double> d2(nb * n*nb);
    std::vector<double> b2(nb * (n-1)*nb);
    std::vector<double> f2(ldf * nrhs);

    auto D = [=,&d](int64_t i, int64_t j) -> double& { return d[i + j*nb]; };

    std::vector<MKL_INT> iseed = {1, 2, 3, 19};


    std::cout << "Testing accuracy of solution of linear equations system" << std::endl;
    std::cout << "with randomly generated positive definite symmetric" << std::endl;
    std::cout << "block tridiagonal coefficient matrix by calculating" << std::endl;
    std::cout << "ratios of residuals to RHS vectors' norms." << std::endl;
    std::cout << "..." << std::endl;

    std::cout << "Matrices are being generated." << std::endl;
    std::cout << "..." << std::endl;

    // Initializing arrays randomly
    LAPACKE_dlarnv(2, iseed.data(), (n-1)*nb*nb, b.data());
    cblas_dcopy((n-1)*nb*nb, b.data(), 1, b2.data(), 1);
    LAPACKE_dlarnv(2, iseed.data(), nrhs*ldf, f.data());
    cblas_dcopy(nrhs*ldf, f.data(), 1, f2.data(), 1);

    for (int64_t k = 0; k < n; k++) {
        for (int64_t j = 0; j < nb; j++) {
            LAPACKE_dlarnv(2, iseed.data(), nb-j, &D(j, k*nb+j));
            cblas_dcopy(nb-j-1, &D(j+1, k*nb+j), 1, &D(j, k*nb+j+1), nb);
        }
        // Diagonal dominance to make the matrix positive definite
        for (int64_t j = 0; j < nb; j++) {
            D(j, k*nb+j) += nb*3.0;
        }
    }
    cblas_dcopy(nb*nb*n, d.data(), 1, d2.data(), 1);

    // Factor the coefficient matrix
    std::cout <<  "Call Cholesky factorization" << std::endl;
    std::cout << "..." << std::endl;

    try {
        info = dpbltrf(queue, n, nb, d.data(), nb, b.data(), nb);
    } catch(cl::sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to SYCL API:\n" << e.what() << std::endl;
        info = -1;
    }
    if(info) {
        std::cout << "Cholesky factorization failed. INFO = " << info << std::endl;
        return 1;
    } else {
        std::cout << "Cholesky factorization succeeded." << std::endl;
    }

    // Solve the system of equations with factored coefficient matrix
    std::cout <<  "Call solving the system of linear equations" << std::endl;
    std::cout << "..." << std::endl;

    info = dpbltrs(queue, n, nrhs, nb, d.data(), nb, b.data(), nb, f.data(), ldf);
    if(info) {
        std::cout << "Solution failed. INFO= " << info << std::endl;
        return 1;
    } else {
        std::cout << "Solution succeeded." << std::endl;
    }

    // Test the accuracy of the solution
    std::cout <<  "The system is solved. Testing the residual" << std::endl;
    std::cout << "..." << std::endl;
    double res = test_res1(n, nrhs, nb, d2.data(), nb, b2.data(), nb, f2.data(), ldf, f.data(), ldf);
    double eps = LAPACKE_dlamch('E');
    if(res/eps > 10.0) {
        std::cout << "Residual test" << std::endl;
        std::cout <<  "max_(i=1,...,NRHS){||A*X(i)-F(i)||/||F(i)||} <= 10*EPS " << std::endl;
        std::cout << "failed" << std::endl;
        return 1;
    } else {
        std::cout << "Residual test" << std::endl;
        std::cout <<  "max_(i=1,...,NRHS){||A*X(i)-F(i)||/||F(i)||} <= 10*EPS " << std::endl;
        std::cout << "passed" << std::endl;
    }

    return 0;
}
