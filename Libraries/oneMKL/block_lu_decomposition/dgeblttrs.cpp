//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*  Content:
*      Function DGEBLTTRS for solving a system of linear equations 
*      with LU-factored block tridiagonal coefficient matrix and 
*      multiple right hand sides.
************************************************************************
* Definition:
* ===========
*   int64_t dgeblttrs(sycl::queue queue, int64_t n, int64_t nb, int64_t nrhs, double* d, double* dl, double* du1, double* du2, int64_t* ipiv, double* f, int64_t ldf)
*
* Purpose:
* ========  
* DGEBLTTRS solves system of linear equations AX = F with general block 
* tridiagonal coefficient matrix 
*          (D_1  C_1                          )
*          (B_1  D_2  C_2                     )
*          (     B_2  D_3  C_3                )
*      A=  (           .........              )
*          (              B_N-2  D_N-1  C_N-1 ) 
*          (                     B_N-1  D_N   )
*      
* LU-factored by DGEBLTTRF and multiple RHS F.
* 
* Arguments:
* ==========  
* QUEUE (input) sycl queue
*     The device queue
*
* N (input) int64_t
*     The number of block rows of the matrix A.  N > 0.
*
* NB (input) int64_t
*     The size of blocks.  NB > 0.
*
* NRHS (input) int64_t
*     The number of right hand sides. NRHS > 0.
*
* D (input) double array, dimension (NB) * (N*NB)
*     The array stores N diagonal blocks (each of size NB by NB) 
*         of triangular factors L and U as they are returned by 
*         DGEBLTTRF. Diagonal blocks of factors L and U are lower and 
*         upper triangular respectively, and their diagonal blocks 
*         with the same index are stored in a block of D with the same
*         index occupying  respectively lower and upper triangles of a
*         block in D. Unit diagonal elements of factor L are not stored. 
*
* DL (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores subdiagonal blocks of lower triangular factor L 
*     as they are returned by DGEBLTTRF.
*      
* DU1 (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores superdiagonal blocks of upper triangular factor U 
*     as they are returned by DGEBLTTRF.
*
* DU2 (input) double array, dimension (NB) * ((N-2)*NB)
*     The array stores blocks of the second superdiagonal of upper 
*          triangular factor U as they are returned by DGEBLTTRF.
*
* IPIV (input) int64_t array, dimension (NB) * (N)
*     The array stores pivot 'local' row indices ('local' means indices
*         vary in the range 1..2*NB. Global row index is 
*         IPIV(I,K) + (K-1)*NB ).
*
* F (input/output) double array, dimension (LDF) * (NRHS)
*     On entry, the array stores NRHS columns of right hand F of the
*         system of linear equations AX = F.
*     On exit, the array stores NRHS columns of unknowns  of the system
*         of linear equations AX = F.
*
* LDF (input) int64_t. LDF >= N*NB
*     Leading dimension of the array F
*
* INFO (return) int64_t
*     = 0:        successful exit
*     < 0:        if INFO = -i, the i-th argument had an illegal value
***********************************************************************/
#include <cstdint>
#include <CL/sycl.hpp>
#include "mkl.h"

#if __has_include("oneapi/mkl.hpp")
#include "oneapi/mkl.hpp"
#else
// Beta09 compatibility -- not needed for new code.
#include "mkl_sycl.hpp"
#endif

using namespace oneapi;

int64_t dgeblttrs(sycl::queue queue, int64_t n, int64_t nb, int64_t nrhs, double* d, double* dl, double* du1, double* du2, int64_t* ipiv, double* f, int64_t ldf) {
    
    // Matrix accessors
    auto D    = [=,&d]    (int64_t i, int64_t j) -> double&  { return  d[i + j*nb];    };
    auto DL   = [=,&dl]   (int64_t i, int64_t j) -> double&  { return  dl[i + j*nb];   };
    auto DU1  = [=,&du1]  (int64_t i, int64_t j) -> double&  { return  du1[i + j*nb];  };
    auto DU2  = [=,&du2]  (int64_t i, int64_t j) -> double&  { return  du2[i + j*nb];  };
    auto IPIV = [=,&ipiv] (int64_t i, int64_t j) -> int64_t& { return  ipiv[i + j*nb]; };
    auto F    = [=,&f]    (int64_t i, int64_t j) -> double&  { return  f[i + j*ldf];   };

    //    Test the input arguments.
    int64_t info = 0;
    if (n <= 0) 
        info = -1;
    else if (nb <= 0)
        info = -2;
    else if (nrhs <= 0)
        info = -3;
    else if (ldf < n*nb)
        info = -10;

    if (info)
        return info;
    
    sycl::context context = queue.get_context();
    sycl::device device = queue.get_device();

    // Forward substitution
    // In the loop compute components Y_K stored in array F
    for (int64_t k = 0; k < n-2; k++) {
        for (int64_t i = 0; i < nb; i++) { 
            if (IPIV(i,k) != i+1){
                auto event1 = mkl::blas::swap(queue, nrhs, &F(k*nb+i, 0), ldf, &F(k*nb+IPIV(i,k)-1, 0), ldf);
                event1.wait_and_throw();
            }
        }
        auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, nb, nrhs, 1.0, &D(0, k*nb), nb, &F(k*nb, 0), ldf);
        auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &DL(0, k*nb), nb, &F(k*nb, 0), ldf, 1.0, &F((k+1)*nb, 0), ldf, {event1});
        event2.wait_and_throw();
    }

    // Apply two last pivots      
    for (int64_t i = 0; i < nb; i++) {
        if (IPIV(i,n-2) != i+1){
            auto event1 = mkl::blas::swap(queue, nrhs, &F((n-2)*nb+i, 0), ldf, &F((n-2)*nb+IPIV(i,n-2)-1, 0), ldf);
            event1.wait_and_throw();
        }
    }

    for (int64_t i = 0; i < nb; i++) {
        if (IPIV(i,n-1) != i+1 + nb){
            auto event1 = mkl::blas::swap(queue, nrhs, &F((n-1)*nb+i, 0), ldf, &F((n-2)*nb+IPIV(i,n-1)-1, 0), ldf);
            event1.wait_and_throw();
        }
    }

    // Computing components Y_N-1 and Y_N out of loop      
    {
        auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, nb, nrhs, 1.0, &D(0, (n-2)*nb), nb, &F((n-2)*nb, 0), ldf);
        auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &DL(0, (n-2)*nb), nb, &F((n-2)*nb, 0), ldf, 1.0, &F((n-1)*nb, 0), ldf, {event1});
        auto event3 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, nb, nrhs, 1.0, &D(0, (n-1)*nb), nb, &F((n-1)*nb, 0), ldf, {event2});
        event1.wait_and_throw();
    }

    // Backward substitution      
    // Computing _N out of loop and store in array F
    {
        auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, (n-1)*nb), nb, &F((n-1)*nb, 0), ldf);
        event1.wait_and_throw();
    }

    // Computing _N-1 out of loop and store in array F
    {
        auto event1 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &DU1(0, (n-2)*nb), nb, &F((n-1)*nb, 0), ldf, 1.0, &F((n-2)*nb, 0), ldf);
        auto event2 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, (n-2)*nb), nb, &F((n-2)*nb, 0), ldf, {event1});
        event2.wait_and_throw();
    }

    // In the loop computing components _K stored in array F
    for (int64_t k = n-3; k >= 0; k--) {
        auto event1 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &DU1(0, k*nb), nb, &F((k+1)*nb, 0), ldf, 1.0, &F(k*nb, 0), ldf);
        auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, nb, nrhs, nb, -1.0, &DU2(0, k*nb), nb, &F((k+2)*nb, 0), ldf, 1.0, &F(k*nb, 0), ldf, {event1});
        auto event3 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::upper, mkl::transpose::nontrans, mkl::diag::nonunit, nb, nrhs, 1.0, &D(0, k*nb), nb, &F(k*nb, 0), ldf, {event2});
        event3.wait_and_throw();
    }

    return 0; 
}
