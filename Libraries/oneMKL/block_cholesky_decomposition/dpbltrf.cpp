//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*      Function DPBLTRF for Cholesky factorization of symmetric
*         positive definite block tridiagonal matrix.
************************************************************************/
#include <CL/sycl.hpp>
#include <cstdint>
#include "mkl.h"

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

using namespace oneapi;

/************************************************************************
* Definition:
* ===========
*   int64_t dpbltrf(cl::sycl::queue queue, int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb) {
*
* Purpose:
* ========
* DPBLTRF computes Cholesky L*L^t-factorization of symmetric positive
* definite block tridiagonal matrix A
*   D_1  B_1^t
*   B_1  D_2   B_2^t
*        B_2  D_3   B_3^t
*           .     .      .
*               .     .      .
*                 B_N-2  D_N-1   B_N-1^t
*                        B_N-1    D_N
* The factorization has the form A = L*L**t, where L is a lower
* bidiagonal block matrix
*   L_1
*   C_1  L_2
*        C_2   L_3
*           .     .      .
*               .     .      .
*                 C_N-2  L_N-1
*                        C_N-1    L_N
* This is a block version of LAPACK DPTTRF subroutine.
*
* Arguments:
* ==========
* QUEUE (input) sycl queue
*     The device queue
*
* N (input) int64_t
*     The number of block rows of the matrix A.  N >= 0.
*
* NB (input) int64_t
*     The size of blocks.  NB >= 0.
*
* D (input/output) double array, dimension (LDD)*(N*NB)
*     On entry, the array stores N diagonal blocks (each of size NB by
*         NB) of the matrix to be factored. The blocks are stored
*         sequentially: first NB columns of D store block D_1, second NB
*         columns store block D_2,...,last NB columns store block D_N.
*     Note: As the diagonal blocks are symmetric only lower or upper
*     ====
*         triangle is needed to store blocks' elements. In this code
*         lower storage is used***
*     On exit, the array stores diagonal blocks of triangular factor L.
*         Diagonal blocks of lower triangular factor L replace
*         respective lower triangles of blocks D_j (1 <= j <= N).
*     Caution: upper triangles of diagonal blocks are not zeroed on exit
*
* LDD (input) int64_t
*     The leading dimension of array D. LDD >= NB.
*
* B (input/output) double array, dimension (LDB)*((N-1)*NB)
*     On entry, the array stores sub-diagonal  blocks (each of size NB
*         by NB) of the matrix to be factored. The blocks are stored
*         sequentially: first NB columns of B store block B_1, second
*         NB columns store block B_2,...,last NB columns store block
*         B_N-1.
*     On exit, the array stores sub-diagonal blocks of triangular factor
*         L.
*
* LDB (input) int64_t
*     The leading dimension of array B. LDB >= NB.
*
* INFO (return) int64_t
*     = 0:        successful exit
*     < 0:        if INFO = -i, the i-th argument had an illegal value
*     > 0:        if INFO = i, the leading minor of order i (and
*                 therefore the matrix A itself) is not
*                 positive-definite, and the factorization could not be
*                 completed. This may indicate an error in forming the
*                 matrix A.
***********************************************************************/
int64_t dpbltrf(cl::sycl::queue queue, int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb) {

    // Matrix accessors
    auto D = [=](int64_t i, int64_t j) -> double& { return d[(i) + (j)*ldd]; };
    auto B = [=](int64_t i, int64_t j) -> double& { return b[(i) + (j)*ldb]; };

    int64_t info = 0;
    if (n < 0)
        info = -1;
    else if (nb < 0)
        info = -2;
    else if (ldd < nb)
        info = -4;
    else if (ldb < nb)
        info = -6;

    if (info)
        return info;

    cl::sycl::context context = queue.get_context();
    cl::sycl::device device = queue.get_device();

    // Compute Cholesky factorization of the first diagonal block
    try {
        std::int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue, mkl::uplo::lower, nb, ldd);
        double* scratchpad = static_cast<double*>(sycl::malloc_shared(scratchpad_size * sizeof(double), device, context));
        if (scratchpad_size != 0 && !scratchpad) {
            info = -1000;
            goto cleanup;
        }
        auto event1 = mkl::lapack::potrf(queue, mkl::uplo::lower, nb, d, ldd, scratchpad, scratchpad_size );
        event1.wait_and_throw();
        sycl::free(scratchpad, context);
    } catch(mkl::lapack::exception const& e) {
        // Handle LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
        if (e.info() > 0) {
        // INFO is equal to the 'global' index of the element u_ii of the factor
        // U which is equal to zero
            info = e.info();
        }
        return info;
    }

    // Main loop
    for (int64_t k = 0; k < n-1; k++) {
        auto event1 = mkl::blas::trsm(queue, mkl::side::right, mkl::uplo::lower, mkl::transpose::trans,
                mkl::diag::nonunit, nb, nb, 1.0, &D(0,k*nb), ldd, &B(0,k*nb), ldb);
        auto event2 = mkl::blas::syrk(queue, mkl::uplo::lower, mkl::transpose::nontrans, nb, nb,
                -1.0, &B(0,k*nb), ldb, 1.0, &D(0,(k+1)*nb), ldd, {event1});
        event2.wait_and_throw();

        try {
            std::int64_t scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue, mkl::uplo::lower, nb, ldd);
            double* scratchpad = static_cast<double*>(sycl::malloc_shared(scratchpad_size * sizeof(double), device, context));
            if (scratchpad_size != 0 && !scratchpad) {
                info = -1000;
                goto cleanup;
            }
            auto event1 = mkl::lapack::potrf(queue, mkl::uplo::lower, nb, &D(0,(k+1)*nb), ldd, scratchpad, scratchpad_size );
            event1.wait_and_throw();
            sycl::free(scratchpad, context);
        } catch(mkl::lapack::exception const& e) {
            // Handle LAPACK related exceptions happened during synchronous call
            std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
            if (e.info() > 0) {
            // INFO is equal to the 'global' index of the element u_ii of the factor
            // U which is equal to zero
                info = e.info() + (k+1)*nb;
            }
            return info;
        }
    }
    return info;

cleanup:
    return info;
}
