//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*
*  Content:
*      Function DGEBLTTRF for LU factorization of general block 
*         tridiagonal matrix;
*      Function PTLDGETRF for partial LU factorization of general 
*         rectangular matrix.
************************************************************************/
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

int64_t ptldgetrf(sycl::queue queue, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, int64_t* ipiv);

/************************************************************************
* Definition:
* ===========
*   int64_t dgeblttrf(sycl::queue queue, int64_t n, int64_t nb, double* d, double* dl, double* du1, double* du2, int64_t* ipiv) {
*
* Purpose:
* ========  
* DGEBLTTRF computes LU factorization of general block tridiagonal 
* matrix
*          (D_1  C_1                          )
*          (B_1  D_2  C_2                     )
*          (     B_2  D_3  C_3                )
*          (           .........              )
*          (              B_N-2 D_N-1  C_N-1  )
*          (                    B_N-1  D_N    )
* using elimination with partial pivoting and row interchanges. 
* The factorization has the form A = L*U, where L is a product of 
* permutation and unit lower bidiagonal block matrices and U is upper 
* triangular with nonzeroes in only the main block diagonal and first 
* two block superdiagonals.  
* This is a block version of LAPACK DGTTRF subroutine.
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
* D (input/output) double array, dimension (NB)*(N*NB)
*     On entry, the array stores N diagonal blocks (each of size NB by NB) 
*         of the matrix to be factored. The blocks are stored 
*         sequentially: first NB columns of D store block D_1, second NB 
*         columns store block D_2,...,last NB columns store block D_N.
*     On exit, the array stores diagonal blocks of triangular factor L 
*         and U. Diagonal blocks of lower triangular factor L replace
*         respective lower triangles of blocks D_j (1 <= j <= N). 
*         Diagonal units are not stored. Diagonal blocks of upper 
*         triangular factor U replace respective upper triangles of 
*         blocks D_j.
*
* DL (input/output) double array, dimension (NB)*((N-1)*NB)
*     On entry, the array stores N-1 subdiagonal blocks (each of size  
*         NB by NB) of the matrix to be factored. The blocks are stored 
*         sequentially: first NB columns of DL store block B_1, second 
*         NB columns store block B_2,...,last NB columns store block
*         B_N-1.      
*     On exit, the array stores subdiagonal blocks of lower triangular  
*         factor L.
*
* DU1 (input/output) double array, dimension (NB)*((N-1)*NB)
*     On entry, the array stores N-1 superdiagonal blocks (each of size  
*         NB by NB) of the matrix to be factored. The blocks are stored 
*         sequentially: first NB columns of DU1 store block C_1, second  
*         NB columns store block C_2,...,last NB columns store block 
*         C_N-1.
*     On exit, the array stores superdiagonal blocks of triangular  
*         factor U.
*
* DU2 (output) double array, dimension (NB)*((N-2)*NB)
*     On exit, the array stores blocks of the second superdiagonal of   
*         triangular factor U.
*
* IPIV (output) int64_t array, dimension (NB)*(N)
*     The pivot 'local' row indices ('local' means indices vary in the 
*     range 1..2*NB. Global row index is IPIV(I,K) + (K-1)*NB ).
*
* INFO (return) int64_t
*     = 0:        successful exit
*     = -1000     memory buffer could not be allocated
*     < 0:        if INFO = -i, the i-th argument had an illegal value
*     > 0:        if INFO = i, U(i,i) is exactly zero. The factorization
*                 can be not completed. 
***********************************************************************/
int64_t dgeblttrf(sycl::queue queue, int64_t n, int64_t nb, double* d, double* dl, double* du1, double* du2, int64_t* ipiv) {

    // Matrix accessors
    auto D    = [=,&d]    (int64_t i, int64_t j) -> double&  { return  d[i + j*nb];    };
    auto DL   = [=,&dl]   (int64_t i, int64_t j) -> double&  { return  dl[i + j*nb];   };
    auto DU1  = [=,&du1]  (int64_t i, int64_t j) -> double&  { return  du1[i + j*nb];  };
    auto DU2  = [=,&du2]  (int64_t i, int64_t j) -> double&  { return  du2[i + j*nb];  };
    auto IPIV = [=,&ipiv] (int64_t i, int64_t j) -> int64_t& { return  ipiv[i + j*nb]; };
    
    // Test the input arguments.
    int64_t info=0;
    if(n <= 0) 
        info = -1;
    else if(nb <= 0) 
        info = -2;
    if(info) 
        return info;

    sycl::context context = queue.get_context();
    sycl::device device = queue.get_device();

    // Allocating a contiguous USM array for partial factorizations
    const int64_t lda = 2*nb;
    double* a = sycl::malloc_shared<double>(lda * 3*nb, device, context);
    auto A = [=,&a](int64_t i, int64_t j) -> double& { return a[i + j*lda]; };
    if (!a) {
        info = -1000;
        goto cleanup;
    }


    for (int64_t k = 0; k < n-2; k++){
        // Form a 2*NB x 3*NB submatrix
        //     D_K   C_K 0
        //     B_K D_K+1 C_K+1
        for (int64_t j = 0; j < nb; j++) {
            auto event1 = mkl::blas::copy(queue, nb,   &D(0,(k)*nb + j), 1,   &A(0,     j),1);
            auto event2 = mkl::blas::copy(queue, nb,  &DL(0,(k)*nb + j), 1,   &A(nb,    j),1);
            auto event3 = mkl::blas::copy(queue, nb, &DU1(0,(k)*nb + j), 1,   &A(0,  nb+j),1);
            auto event4 = mkl::blas::copy(queue, nb,   &D(0,(k+1)*nb + j),     1,   &A(nb,  nb+j),1);
            auto event5 = mkl::blas::copy(queue, nb, &DU1(0,(k+1)*nb + j),     1,   &A(nb,2*nb+j),1);
            event1.wait_and_throw();
            event2.wait_and_throw();
            event3.wait_and_throw();
            event4.wait_and_throw();
            event5.wait_and_throw();

            queue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for(sycl::range<1>(nb), [=] (sycl::id<1> it) {
                            const int64_t i = it[0];
                            a[i + (2*nb + j)*lda] = 0.0;
                            });
                    });
            queue.wait();

        }



        // Partial factorization of the submatrix
        //     (D_K    C_K   0    )        (L_K,K    )   (U_K,K U_K,K+1, U_K,K+2)
        //     (                  )  = P * (         ) *                          
        //     (B_K  D_K+1   C_K+1)        (L_K+1,K+1)                            
        //
        //    (  0    0       0     )
        //  + (                     )
        //    (  0    D'_K+1  C'_K+1)
        info = ptldgetrf(queue, 2*nb, 3*nb, nb, &A(0,0), lda, &IPIV(0,k));
        if (info > 0) {
        // INFO is equal to the 'global' index of the element u_ii of the factor 
        // U which is equal to zero
            return info + k*nb;
        }

        // Factorization results to be copied back to arrays:
        // L_K,K, U_K,K, D'_K+1 -> D
        // L_K+1,K -> DL
        // U_K,K+1 -> DU1
        // U_K,K+2 -> DU2
        for(int64_t j = 0; j < nb; j++) {
            auto event1 = mkl::blas::copy(queue, nb, &A( 0,     j), 1,   &D(0,k*nb + j), 1);
            auto event2 = mkl::blas::copy(queue, nb, &A(nb,     j), 1,  &DL(0,k*nb + j), 1);
            auto event3 = mkl::blas::copy(queue, nb, &A( 0,  nb+j), 1, &DU1(0,k*nb + j), 1);
            auto event4 = mkl::blas::copy(queue, nb, &A(nb,  nb+j), 1,   &D(0,(k+1)*nb     + j), 1);
            auto event5 = mkl::blas::copy(queue, nb, &A( 0,2*nb+j), 1, &DU2(0,k*nb + j), 1);
            auto event6 = mkl::blas::copy(queue, nb, &A(nb,2*nb+j), 1, &DU1(0,(k+1)*nb     + j), 1);
            event1.wait_and_throw();
            event2.wait_and_throw();
            event3.wait_and_throw();
            event4.wait_and_throw();
            event5.wait_and_throw();
            event6.wait_and_throw();
        }
    }

    // Out of loop factorization of the last 2*NBx2*NB submatrix
    //  (D_N-1    C_N-1)          (L_N-1,N-1      0)   (U_N-1,N-1   U_N-1,N )
    //  (              ) = P_N-1* (                ) * (                    )
    //  (B_N-1      D_N)          (  L_N,N-1  L_N,N)   (      0     U_N,N   )
    for(int64_t j = 0; j < nb; j++) {
        auto event1 = mkl::blas::copy(queue, nb,   &D(0, (n-2)*nb + j), 1, &A(0,    j), 1);
        auto event2 = mkl::blas::copy(queue, nb,  &DL(0, (n-2)*nb + j), 1, &A(nb+0,    j), 1);
        auto event3 = mkl::blas::copy(queue, nb, &DU1(0, (n-2)*nb + j), 1, &A(   0, nb+j), 1);
        auto event4 = mkl::blas::copy(queue, nb,   &D(0, (n-1)*nb + j), 1, &A(nb+0, nb+j), 1);
        event1.wait_and_throw();
        event2.wait_and_throw();
        event3.wait_and_throw();
        event4.wait_and_throw();
    }

    // Pivoting array for the last factorization has 2*NB elements stored in
    // two last columns of IPIV
    try {
        std::int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, 2*nb, 2*nb, lda);
        double* scratchpad = sycl::malloc_shared<double>(scratchpad_size, device, context);
        if (!scratchpad) {
            info = -1000;
            goto cleanup;
        }
        auto event1 = mkl::lapack::getrf(queue, 2*nb, 2*nb, &A(0,0), lda, &IPIV(0,n-2), scratchpad, scratchpad_size );
        event1.wait_and_throw();
        sycl::free(scratchpad, context);
    } catch(mkl::lapack::exception const& e) {
        // Handle LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
        if (e.info() > 0) {
        // INFO is equal to the 'global' index of the element u_ii of the factor  
        // U which is equal to zero
            info = e.info() + (n-2)*nb;
        }
        return info;
    }

    // Copy the last result back to arrays:
    // L_N-1,N-1, L_N,N, U_N-1,N-1, U_N,N  -> D
    // L_N,N-1  -> DL
    // U_N-1,N  -> DU1
    for (int64_t j = 0; j < nb; j++) {
        auto event1 = mkl::blas::copy(queue, nb,    &A(0,      j), 1,   &D(0, (n-2)*nb + j), 1);
        auto event2 = mkl::blas::copy(queue, nb, &A(nb+0,      j), 1,  &DL(0, (n-2)*nb + j), 1);
        auto event3 = mkl::blas::copy(queue, nb,    &A(0, nb + j), 1, &DU1(0, (n-2)*nb + j), 1);
        auto event4 = mkl::blas::copy(queue, nb, &A(nb+0, nb + j), 1,   &D(0, (n-1)*nb + j), 1);
        event1.wait_and_throw();
        event2.wait_and_throw();
        event3.wait_and_throw();
        event4.wait_and_throw();
    }

    sycl::free(a, context);
    return info;

cleanup:
    sycl::free(a, context);
    return info;

}



/**********************************************************************
* Purpose:
* ========  
* PTLDGETRF computes partial (in a case K<min(M,N)) LU factorization 
* of matrix A = P*(L*U+A1)
*
* Arguments:
* ==========     
*  M (input) int64_t
*     The number of rows of the matrix A.  M >= 0.
*
*  N (input) int64_t
*     The number of columns of the matrix A.  N >= 0.
*     
*  K (input) int64_t
*     The number of columns of the matrix A participating in 
*     factorization. N >= K >= 0
*
*  A (input/output) double array, dimension (LDA)*(N)
*     On entry, the M-by-N matrix A to be factored.
*     On exit:
*         if K >= min(M,N), A is overwritten by details of its LU
*                 factorization as returned by DGETRF.
*         if K < min(M,N), partial factorization A = P * (L * U + A1) 
*             is performed where P is permutation matrix (pivoting);
*         L is M by K lower trapezoidal (with unit diagonal) matrix  
*             stored in lower MxK trapezoid of A. Diagonal units 
*                 are not stored.
*         U is K by N upper trapezoidal matrix stored in upper 
*             K by N trapezoid of A;
*         A1 is (M-K) by (N-K) residual stored in intersection
*             of last M-K rows and last N-K columns of A.
*
*  LDA (input) int64_t
*     The leading dimension of the array A.  LDA >= max(1,M).
*
*  IPIV (output) int64_t array, dimension (min(M,K))
*     The pivot indices; for 1 <= i <= min(M,K), row i of the
*     matrix was interchanged with row IPIV(i).
*
*  INFO (return) int64_t
*     = 0:  successful exit
*     < 0:  if INFO = -i, the i-th argument had an illegal value
*     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
*           can be not completed. 
***********************************************************************/
#undef    A
#undef    D
#undef   DL
#undef  DU1
#undef  DU2
#undef IPIV
int64_t ptldgetrf(sycl::queue queue, int64_t m, int64_t n, int64_t k, double* a, int64_t lda, int64_t* ipiv) {

    auto A = [=,&a](int64_t i, int64_t j) -> double& { return a[i + j*lda]; };

    sycl::context context = queue.get_context();
    sycl::device device = queue.get_device();

    int64_t info=0;
    if(m < 0)
        info = -1;
    else if(n < 0)
        info = -2;
    else if( (k > n) || (k <0))
        info = -3;
    else if(lda < m)
        info = -5;

    if(info)
        return info;

    if(k < std::min<int64_t>(m,n)) {

        // LU factorization of first K columns
        {
            try {
                std::int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, m, k, lda);
                double* scratchpad = sycl::malloc_shared<double>(scratchpad_size, device, context);
                if (!scratchpad) {
                    info = -1000;
                    return info;
                }
                auto event1 = mkl::lapack::getrf(queue, m, k, &A(0,0), lda, &ipiv[0], scratchpad, scratchpad_size );
                event1.wait_and_throw();
                sycl::free(scratchpad, context);
            } catch(mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during synchronous call
                std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\ninfo: " << e.info() << std::endl;
                if (e.info() > 0) {
                    info = e.info();
                }
                return info;
            }
        }
        for (int64_t i = 0; i < k; i++) {
            if(ipiv[i] != i+1) {
                // Applying permutations returned by DGETRF to last N-K columns
                auto event1 = mkl::blas::swap(queue, n-k, &A(i,k), lda, &A(ipiv[i]-1, k), lda);
                event1.wait_and_throw();
            }
        }
        // Updating A1
        {
            auto event1 = mkl::blas::trsm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans, mkl::diag::unit, k, n-k, 1.0, &A(0,0), lda, &A(0,k), lda);
            auto event2 = mkl::blas::gemm(queue, mkl::transpose::nontrans, mkl::transpose::nontrans, m-k, n-k, k, -1.0, &A(k,0), lda, &A(0,k), lda, 1.0, &A(k,k), lda, {event1});
            event2.wait_and_throw();
        }
    }
    else {
            std::int64_t scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, m, n, lda);
            double* scratchpad = sycl::malloc_shared<double>(scratchpad_size, device, context);
            if (!scratchpad) {
                info = -1000;
                return info;
            }
            auto event1 = mkl::lapack::getrf(queue, m, n, &A(0,0), lda, &ipiv[0], scratchpad, scratchpad_size );
            event1.wait_and_throw();
            sycl::free(scratchpad, context);
    }

    return info;
}
