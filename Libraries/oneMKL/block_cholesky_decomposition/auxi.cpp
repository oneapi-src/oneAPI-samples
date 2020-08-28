//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*  Content:
*      Auxiliary subroutines for:
*      - Testing accuracy of Cholesky factorization by computing ratio
*        ||A-L*L^t||_F/||A||_F of Frobenius norms of the residual to the
* 	   Frobenius norm of the initial matrix and comparing it to 5*EPS.
*      - Calculating max_(i=1,...,NRHS){||AX(i)-F(i)||/||F(i)||} of
*        ratios of residuals to norms of RHS vectors for a system of
*        linear equations with tridiagonal coefficient matrix and
*        multiple RHS
*
***********************************************************************/
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "mkl.h"


/************************************************************************
* Definition:
* ===========
*   double test_res(int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* d1, int64_t ldd1, double* b1, int64_t ldb1, double* d2, int64_t ldd2, double* b2, int64_t ldb2)
*
* Purpose:
* ========
* Given L*L^t factorization of block tridiagonal matrix A TEST_RES
* computes ratio ||A-L*L^t||_F/||A||_F of Frobenius norm of the residual
* to the Frobenius norm of the initial matrix. The test is considered as
* passed if the ratio does not exceed 5*EPS. The result is returned via
* value of INFO.
*
* Arguments:
* ==========
* N (input) int64_t
*     The number of block rows of the matrix A.  N > 0.
*
* NB (input) int64_t
*     The size of blocks.  NB > 0.
*
* D (input) double array, dimension (LDD) * (N*NB)
*     The array stores N diagonal blocks (each of size NB by NB)
*         of  the triangular factor L if factorized matrix. The blocks
*         are stored sequentially block by block.
*     Caution: upper triangles of diagonal blocks are not zeroed*
*     =======
*
* LDD (input) int64_t.
*     The leading dimension of the array D, LDD >= NB
*
* B (input) double array, dimension (LDB,(N-1)*NB)
*     The array stores sub-diagonal blocks of triangular factor L.
*     The blocks are stored sequentially block by block.
*
* LDB (input) int64_t.
*     The leading dimension of the array B, LDB >= NB
*
* D1 (work array) double array, dimension (LDD1,N*NB)
*     The array is destined for internal computations.
*
* LDD1 (input) int64_t.
*     The leading dimension of the array D1, LDD1 >= NB
*
* B1 (work array) double array, dimension (LDB1,(N-1)*NB)
*     The array is destined for internal computations.
*
* LDB1 (input) int64_t.
*     The leading dimension of the array B1, LDB1 >= NB
*
* D2 (input) double array, dimension (LDD2,N*NB)
*     The array stores N diagonal blocks (each of size NB by NB)
*         of  the initial symmetric positive definite matrix A.
*         The blocks are stored sequentially block by block. The
*         array is used for comparison.
*
* LDD2 (input) int64_t.
*     The leading dimension of the array D2, LDD2 >= NB
*
* B2 (input) double array, dimension (LDB2,(N-1)*NB)
*     The array stores sub-diagonal blocks of the initial symmetric
*         positive definite matrix A. The blocks are stored
*         sequentially block by block. The array is used for comparison.
*
* LDB2 (input) int64_t.
*     The leading dimension of the array B2, LDB2 >= NB
*
* INFO (output) int64_t
*     = 0:        successful exit
*     < 0:        if INFO = -i, the i-th argument had an illegal value
*     = 1:        the ratio ||A-L*L^t||_F/||A||_F exceeds 5*EPS
***********************************************************************/

double test_res(int64_t n, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* d1, int64_t ldd1, double* b1, int64_t ldb1, double* d2, int64_t ldd2, double* b2, int64_t ldb2) {

    // Matrix accessors
    auto D  = [=](int64_t i, int64_t j) -> double& { return d[i + j*ldd];   };
    auto D1 = [=](int64_t i, int64_t j) -> double& { return d1[i + j*ldd1]; };
    auto D2 = [=](int64_t i, int64_t j) -> double& { return d2[i + j*ldd2]; };
    auto B  = [=](int64_t i, int64_t j) -> double& { return b[i + j*ldb];   };
    auto B1 = [=](int64_t i, int64_t j) -> double& { return b1[i + j*ldb1]; };
    auto B2 = [=](int64_t i, int64_t j) -> double& { return b2[i + j*ldb2]; };

    // Compute S2 = ||A||_F
    double s  = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*n, d2, ldd2);
    double s1 = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*(n-1), b2, ldb2);
    double s2 = sqrt(s*s+2.0*s1*s1);

    // Copy D -> D1, B -> B1 and nullify the upper triangle of blocks in D1
    for (int64_t k = 0; k < n; k++) {
        for(int64_t j = 0; j < nb; j++) {
            cblas_dcopy(nb-j, &D(j, k*nb+j), 1, &D1(j, k*nb+j), ldd1);
            for (int64_t i = 0; i < j; i++) {
                D1(j, k*nb+i) = 0.0;
            }
        }
    }

    for (int64_t k = 0; k < n-1; k++) {
        for (int64_t j = 0; j < nb; j++) {
            cblas_dcopy(nb, &B(0, k*nb+j), 1, &B1(0, k*nb+j), 1);
        }
    }

    // Compute product of lower block bidiagonal matrix by its transpose
    // | L_1                        | | L_1^t B_1^t                      |
    // | B_1 L_2                    | |       L_2^t B_2^t                |
    // |    .    .                  |*|           .       .              |
    // |        .    .              | |               .        .         |
    // |          B_N-2 L_N-1       | |                  L_N-1^t  B_N-1^t|
    // |                B_N-1 L_N | |                             L_N^t  |
    //
    // Result matrix has the following structure
    //   D_1  B_1^t
    //   B_1  D_2   B_2^t
    //        B_2   D_3   B_3^t
    //           .     .      .
    //               .     .      .
    //                 B_N-2  D_N-1   B_N-1^t
    //                        B_N-1    D_N
    //
    // D_1 := L_1*L_1^t
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, nb, nb, 1.0, d, ldd, d1, ldd1);
        for (int64_t k = 0; k < n-1; k++) {
            // B_k := B_k*L_k^t
            cblas_dtrmm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    nb, nb, 1.0, &D(0, k*nb), ldd, &B1(0, k*nb), ldb1);
            // D_k := L_k*L_k^t
            cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                    nb, nb, 1.0, &D(0, (k+1)*nb), ldd, &D1(0, (k+1)*nb), ldd1);
            // D_k := D_k + B_k*B_k^t
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nb, nb, nb, 1.0,
                    &B(0, k*nb), ldb, &B(0, k*nb), ldb, 1.0,
                    &D1(0, (k+1)*nb), ldd1);
        }

    // Compute the difference between the calculated product L*L^t and initial
    // matrix that was factored
    for (int64_t j = 0; j < nb*n; j++) {
        for (int64_t i = 0; i < nb; i++) {
            D1(i,j) = D1(i,j) - D2(i,j);
        }
    }

    for (int64_t j = 0; j < nb*(n-1); j++) {
        for (int64_t i = 0; i < nb; i++) {
            B1(i,j) = B1(i,j) - B2(i,j);
        }
    }

    s  = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*n, d1, ldd1);
    s1 = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*(n-1), b1, ldb1);

    s = sqrt(s*s+2.0*s1*s1)/s2;
    return s;
}

/************************************************************************
* Definition:
* ===========
*   double test_res1(int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf, double* x, int64_t ldx )
*
* Purpose:
* ========
* Given approximate solution X of system of linear equations A*X=F
* with symmetric positive definite block tridiagonal coefficient matrix
* A =
*   D_1  B_1^t
*   B_1  D_2   B_2^t
*        B_2  D_3   B_3^t
*           .     .      .
*               .     .      .
*                 B_N-2  D_N-1   B_N-1^t
*                        B_N-1    D_N
* the routine computes max_(i=1,...,NRHS){||AX(i)-F(i)||/F(i)} of ratios
* of residuals to norms of RHS vectors. he test is considered as passed
* if the value does not exceed 10*EPS  where EPS is the machine
* precision.
*
* Arguments:
* ==========
* N (input) int64_t
*     The number of block rows of the matrix A.  N > 0.
*
* NRHS (input) int64_t
*     The number of right hand sides (number of columns in matrix F.
*
* NB (input) int64_t
*     The block size of blocks D_j, B_j
*
* D (input) double array, dimension (LDD) * (N*NB)
*     The array stores N diagonal blocks (each of size NB by NB)
*         of  matrix A. The blocks are stored sequentially block by
*         block.
*     Caution: The diagonal blocks are symmetric matrices  - this
*     =======
*         feature is assumed.
*
* LDD (input) int64_t.
*     The leading dimension of the array D, LDD >= NB
*
* B (input) double array, dimension (LDB) * ((N-1)*NB)
*     The array stores sub-diagonal blocks of matrix A.
*     The blocks are stored sequentially block by block.
*
* LDB (input) int64_t.
*     The leading dimension of the array B, LDB >= NB
*
* F (input) double array, dimension (LDF) * (NRHS)
*     The right hand sides of the system of linear equations.
*
* LDF (input) int64_t.
*     The leading dimension of the array F, LDF >= NB*N
*
* X (input) double array, dimension (LDX) * (NRHS)
*     The solutions of the system of linear equations.
*
* LDX (input) int64_t.
*     The leading dimension of the array X, LDX >= NB*N
*
* INFO (output) int64_t
*     = 0:        successful exit
*     < 0:        if INFO = -i, the i-th argument had an illegal value
*     = 1:        max_(i=1,...,NRHS){||AX(i)-F(i)||/F(i)} exceeds 10*EPS
*     = 10:       note enough memory for internal array
***********************************************************************/

double test_res1(int64_t n, int64_t nrhs, int64_t nb, double* d, int64_t ldd, double* b, int64_t ldb, double* f, int64_t ldf, double* x, int64_t ldx ) {

    // Matrix accessors
    auto D = [=](int64_t i, int64_t j) -> double& { return d[i + j*ldd]; };
    auto B = [=](int64_t i, int64_t j) -> double& { return b[i + j*ldb]; };
    auto F = [=](int64_t i, int64_t j) -> double& { return f[i + j*ldf]; };
    auto X = [=](int64_t i, int64_t j) -> double& { return x[i + j*ldx]; };

    std::vector<double> norms(nrhs);

    // Compute norms of RHS vectors
    for (int64_t i = 0; i < nrhs; i++) {
        norms[i] = cblas_dnrm2(nb*n, &F(0,i), 1);
    }

    // Out-of-loop compute F(1):=F(1)-D(1)*X(1)-B(1)^t*X(2)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, d, ldd, x, ldx, 1.0, f, ldf);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, nb, nrhs, nb, -1.0, b,ldb, &X(nb, 0), ldx, 1.0, f, ldf);

    for (int64_t k = 1; k < n-1; k++) {
        // Compute F(K):=F(K)-B(K-1)*X(K-1)-D(K)*X(K)-B(K)^t*X(K+1)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &B(0, (k-1)*nb), ldb, &X((k-1)*nb, 0), ldx, 1.0, &F(k*nb, 0), ldf);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &D(0,     k*nb), ldd, &X(    k*nb, 0), ldx, 1.0, &F(k*nb, 0), ldf);
        cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, nb, nrhs, nb, -1.0, &B(0,     k*nb), ldb, &X((k+1)*nb, 0), ldx, 1.0, &F(k*nb, 0), ldf);
    }

    // Out-of-loop compute F(N):=F(N)-B(N-1)*X(N-1)-D(N)*X(N)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &B(0, (n-2)*nb), ldb, &X((n-2)*nb, 0), ldx, 1.0, &F((n-1)*nb, 0), ldf);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &D(0, (n-1)*nb), ldd, &X((n-1)*nb, 0), ldx, 1.0, &F((n-1)*nb, 0), ldf);

    // Compute norms of residual vectors divided by norms of RHS vectors
    double res = 0.0;
    for (int64_t i = 0; i < nrhs; i++) {
        double s  = cblas_dnrm2(n*nb, &F(0,i), 1);
        res = std::max<double>(res, s/norms[i]);
    }

    return res;
}
