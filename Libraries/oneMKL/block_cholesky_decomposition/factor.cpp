//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
 *
 *  Content:
 *      Example of Cholesky factorization of a symmetric positive
 *      definite block tridiagonal matrix
 ************************************************************************
 * Purpose:
 * ========
 * Testing accuracy of Cholesky factorization A=
 *      | L_1                  | | L_1^t  C_1^t                   |
 *      | C_1  L_2             | |        L_2^t  C_2^t            |
 *  A = |    .     .           |*|             .       .          |
 *      |        .     .       | |                  .     C_N-1^t |
 *      |           C_N-1  L_N | |                        L_N^t   |
 *
 * of a symmetric positive definite block tridiagonal matrix A
 *  | D_1  B_1^t                           |
 *  | B_1  D_2   B_2^t                     |
 *  |      B_2  D_3   B_3^t                |
 *  |         .     .      .               |
 *  |             .     .      .           |
 *  |               B_N-2  D_N-1   B_N-1^t |
 *  |                      B_N-1   D_N     |
 * by calling TEST_RES which calculates ratio of Frobenius norms
 *      ||A-L*L^t||_F/||A||_F.
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

int64_t dpbltrf(sycl::queue queue, int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb);
double test_res(int64_t, int64_t, double*, int64_t, double*, int64_t, double*, int64_t, double*, int64_t, double*, int64_t, double*, int64_t);

template<typename T>
using allocator_t = sycl::usm_allocator<T, cl::sycl::usm::alloc::shared>;


int main() {

    if (sizeof(MKL_INT) != sizeof(int64_t)) {
        std::cerr << "MKL_INT not 64bit" << std::endl;
        return -1;
    }

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

    sycl::device device{cl::sycl::default_selector{}};
    sycl::queue queue(device, error_handler);
    sycl::context context = queue.get_context();

    if (device.get_info<sycl::info::device::double_fp_config>().empty()) {
        std::cerr << "The sample uses double precision, which is not supported" << std::endl;
        std::cerr << "by the selected device. Quitting." << std::endl;
        return 0;
    }

    allocator_t<double> allocator_d(context, device);

    MKL_INT n = 200;
    MKL_INT nb = 20;

    std::vector<double, allocator_t<double>>  d(nb * n*nb,     allocator_d);
    std::vector<double, allocator_t<double>>  b(nb * (n-1)*nb, allocator_d);
    std::vector<double> d1(nb * n*nb);
    std::vector<double> b1(nb * (n-1)*nb);
    std::vector<double> d2(nb * n*nb);
    std::vector<double> b2(nb * (n-1)*nb);

    std::vector<MKL_INT> iseed = {1, 2, 33, 15};

    auto D = [=,&d](int64_t i, int64_t j) -> double& { return d[i + j*nb]; };

    std::cout << "Testing accuracy of Cholesky factorization\n";
    std::cout << "of randomly generated positive definite symmetric\n";
    std::cout << "block tridiagonal matrix by calculating residual.\n\n";
    std::cout << "Matrix size = " << n << "\n";
    std::cout << "Block  size = " << nb << "\n";
    std::cout << "...\n";
    std::cout << "Matrices are being generated.\n";
    std::cout << "...\n";

    // Initializing arrays randomly
    LAPACKE_dlarnv(2, iseed.data(), (n-1)*nb*nb, b.data());
    cblas_dcopy((n-1)*nb*nb, b.data(), 1, b2.data(), 1);
    for (int64_t k = 0; k < n; k++) {
        for (int64_t j = 0; j < nb; j++) {
            LAPACKE_dlarnv(2, iseed.data(), nb-j, &D(j,k*nb+j));
            cblas_dcopy(nb-j, &D(j+1, k*nb+j), 1, &D(j, k*nb+j+1), nb);
        }
        // Diagonal dominance to make the matrix positive definite
        for (int64_t j = 0; j < nb; j++) {
            D(j, k*nb+j) += nb*3.0;
        }
    }
    cblas_dcopy(n*nb*nb, d.data(), 1, d2.data(), 1);

    std::cout << "Call Cholesky factorization\n";
    std::cout << "...\n";
    try {
        info = dpbltrf(queue, n, nb, d.data(), nb, b.data(), nb);
    } catch(sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to SYCL API:\n" << e.what() << std::endl;
        info = -1;
    }

    if(info) {
        std::cout << "Factorization failed. info = " << info << std::endl;
        return 1;
    } else {
        std::cout << "Cholesky factorization succeeded." << std::endl;
        std::cout << "Testing the residual" << std::endl;
        std::cout << "..." << std::endl;
        double res = test_res(n, nb, d.data(), nb, b.data(), nb, d1.data(), nb, b1.data(), nb, d2.data(), nb, b2.data(), nb);
        double eps = LAPACKE_dlamch('E');

        std::cout << "Residual test" << std::endl;
        std::cout <<  "||A-L*L^t||_F/||A||_F <= 5*EPS..." << std::endl;
        if (res/eps > 5.0) {
            std::cout << "failed: ||A-L*L^t||_F/||A||_F = " << res << std::endl;
            return 1;
        } else {
            std::cout << "passed" << std::endl;
        }
    }

    return 0;
}
