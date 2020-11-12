//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*      Function DPBLTRS for solving a system of linear equations with
*      Cholesky factored symmetric positive definite block tridiagonal
*      coefficient matrix.
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
*   int64_t dpbltrs(cl::sycl::queue queue, int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf)
*
* Purpose:
* ========
* DPBLTRS computes a solution to system of linear equations A*X=F with
* symmetric positive definite block tridiagonal coefficient matrix A
*   D_1  B_1^t
*   B_1  D_2   B_2^t
*        B_2  D_3   B_3^t
*           .     .      .
*               .     .      .
*                 B_N-2  D_N-1   B_N-1^t
*                        B_N-1    D_N
* and multiple right hand sides F. Before call this routine the
* coefficient matrix should factored A=L*L^T by calling DPBLTRF where
* L is a lower block bidiagonal matrix
*   L_1
*   C_1  L_2
*        C_2   L_3
*           .     .      .
*               .     .      .
*                 C_N-2  L_N-1
*                        C_N-1    L_N
* This is a block version of LAPACK DPTTRS subroutine.
*
* Arguments:
* ==========
* QUEUE (input) sycl queue
*     The device queue
*
* N (input) int64_t
*     The number of block rows of the matrix A.  N >= 0.
*
* NRHS (input) int64_t
*     The number of right hand sides (the number of columns in matrix F).
*
* NB (input) int64_t
*     The size of blocks.  NB >= 0.
*
* D (input) double array, dimension (LDD) * (N*NB)
*     On entry, the array stores diagonal blocks of triangular factor L.
*         Diagonal blocks L_j of lower triangular factor L are stored as
*         respective lower triangles of blocks D_j (1 <= j <= N).
*         Caution: upper triangles of D_j are not assumed to be zeroed.
*         =======
*
* LDD (input) int64_t
*     The leading dimension of array D. LDD >= NB.
*
* B (input) double array, dimension (LDB) * ((N-1)*NB)
*     On entry, the array stores sub-diagonal blocks L_j of triangular
*          factor L.
*
* LDB (input) int64_t
*     The leading dimension of array B. LDB >= NB.
*
* F   (input/output) double array, dimension (LDF) * (NRHS)
*     On entry, the columns of the array store vectors F(i) of right
*         hand sides of system of linear equations A*X=F.
*
* LDF (input) int64_t
*     The leading dimension of array F. LDF >= NB*N.
*
* INFO (return) int64_t
*     = 0:        successful exit
*     < 0:        if INFO = -i, the i-th argument had an illegal value
* =====================================================================
*/
int64_t dpbltrs(cl::sycl::queue queue, int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf) {

    auto D = [=](int64_t i, int64_t j) -> double& { return d[i + j*ldd]; };
    auto B = [=](int64_t i, int64_t j) -> double& { return b[i + j*ldb]; };
    auto F = [=](int64_t i, int64_t j) -> double& { return f[i + j*ldf]; };

    //    Test the input arguments.
    int64_t info = 0;
    if (n < 0)
        info = -1;
    else if (nrhs < 0)
        info = -2;
    else if (nb < 0)
        info = -3;
    else if (ldd < nb)
        info = -5;
    else if (ldb < nb)
        info = -7;
    else if (ldf < nb*n)
        info = -9;

    if (info)
        return info;

    // Solving the system of linear equations L*Y=F
    mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, d, ldd, f, ldf);
    for ( int64_t k = 0; k < n-1; k++) {
        mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &B(0, k*nb), ldb, &F(k*nb, 0), ldf, 1.0, &F((k+1)*nb, 0), ldf);
        mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, (k+1)*nb), ldd, &F((k+1)*nb, 0), ldf);
    }
    // ..
    // Solving the system of linear equations L^T*X=Y
    mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::trans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, (n-1)*nb), ldd, &F((n-1)*nb, 0), ldf);
    for ( int64_t k = n-2; k >= 0; k-- ) {
        mkl::blas::gemm(queue, mkl::transpose::trans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &B(0, k*nb), ldb, &F((k+1)*nb, 0), ldf, 1.0, &F(k*nb, 0), ldf);
        mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::trans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, k*nb), ldd, &F(k*nb, 0), ldf);
    }

    return info;
}
