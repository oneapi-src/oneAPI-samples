//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/*
*  Content:
*      Auxiliary subroutines for: 
*      - Computing ratio ||A-L*U||_F/||A||_F of Frobenius norm of the   
* 	 residual to the Frobenius norm of the initial matrix. 
*      - Calculating max_(i=1,...,NRHS){||AX(i)-F(i)||/||F(i)||} of  
*        ratios of residuals to norms of RHS vectors for a system of 
*        linear equations with tridiagonal coefficient matrix and 
*        multiple RHS      
*
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "mkl.h"

/***********************************************************************
* Definition:
* ===========
*   double resid1( int64_t n, int64_t nb, double* dl,  double* d,  double* du1,  double* du2,  int64_t* ipiv,  double* dlcpy, double* dcpy, double* du1cpy) {
*
* Purpose:
* ========  
* Given LU factorization of block tridiagonal matrix A function RESID1  
* returns ratio ||A-L*U||_F/||A||_F of Frobenius norm of the residual  
* to the Frobenius norm of the initial matrix. The ratio provides info
* on how good the factorization is.
* 
* Arguments:
* ==========  
* N (input) int64_t
*     The number of block rows of the matrix A.  N > 0.
*
* NB (input) int64_t
*     The size of blocks.  NB > 0.
*
* DL (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 subdiagonal blocks (each of size NB by NB) of  
*         lower triangular factor L. The blocks are stored sequentially 
*         block by block.
*
* D (input) double array, dimension (NB) * (N*NB)
*     The array stores N diagonal blocks (each of size NB by NB) 
*         of triangular factors L and U. 
*
* DU1 (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 superdiagonal blocks (each of size  
*         NB by NB) of triangular factor U.
*
* DU2 (input) double array, dimension (NB) * ((N-2)*NB)
*     The array stores N-2 blocks of the second superdiagonal of   
*         triangular factor U.
*
* IPIV (input) int64_t array, dimension (NB) * (N)
*     The array stores pivot 'local' row indices. 
*
* DLCPY (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 subdiagonal blocks of initial block 
*         tridiagonal matrix.
*
* DCPY (input) double array, dimension (NB) * (N*NB)
*     The array stores N diagonal blocks of initial block 
*         tridiagonal matrix.
*
* DU1CPY (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 superdiagonal blocks of initial block 
*         tridiagonal matrix.
***********************************************************************/
double resid1( int64_t n, int64_t nb, double* dl,  double* d,  double* du1,  double* du2,  int64_t* ipiv,  double* dlcpy, double* dcpy, double* du1cpy) {

    // Matrix accessors
    auto D      = [=,&d]      (int64_t i, int64_t j) -> double&  { return  d[i + j*nb];      };
    auto DCPY   = [=,&dcpy]   (int64_t i, int64_t j) -> double&  { return  dcpy[i + j*nb];   };
    auto DL     = [=,&dl]     (int64_t i, int64_t j) -> double&  { return  dl[i + j*nb];     };
    auto DLCPY  = [=,&dlcpy]  (int64_t i, int64_t j) -> double&  { return  dlcpy[i + j*nb];  };
    auto DU1    = [=,&du1]    (int64_t i, int64_t j) -> double&  { return  du1[i + j*nb];    };
    auto DU1CPY = [=,&du1cpy] (int64_t i, int64_t j) -> double&  { return  du1cpy[i + j*nb]; };
    auto DU2    = [=,&du2]    (int64_t i, int64_t j) -> double&  { return  du2[i + j*nb];    };
    auto IPIV   = [=,&ipiv]   (int64_t i, int64_t j) -> int64_t& { return  ipiv[i + j*nb];   };

    double s = 0.0;
    double norm = 0.0;

    s    = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb,     nb*n,   dcpy, nb);
    norm = s*s;
    s    = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*(n-1),  dlcpy, nb);
    norm = norm + s*s;
    s    = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb*(n-1), du1cpy, nb);
    norm = sqrt(norm + s*s);

    const int64_t ldb = 2*nb;
    std::vector<double> b(ldb*3*nb);
    std::vector<double> b1(ldb*3*nb);
    
    auto B  = [=,&b] (int64_t i, int64_t j) -> double& { return  b[i + j*ldb]; };
    auto B1 = [=,&b1](int64_t i, int64_t j) -> double& { return b1[i + j*ldb]; };

    for(int64_t j = 0; j < 3*nb; j++ ) {
        for(int64_t i = 0; i < 2*nb; i++ ) {
            B(i,j)  = 0.0;
            B1(i,j) = 0.0;
        }
    }

    const int64_t ldl = 2*nb;
    std::vector<double> l(ldl*nb);
    auto L  = [=,&l] (int64_t i, int64_t j) -> double& { return  l[i + j*ldl]; };

    for(int64_t j = 0; j < nb; j++) {
        for(int64_t i = 0; i < 2*nb; i++) {
            L(i,j) = 0.0;
        }
        L(j,j) = 1.0;
    }

    const int64_t ldu = nb;
    std::vector<double> u(ldu*3*nb);
    auto U  = [=,&u] (int64_t i, int64_t j) -> double& { return  u[i + j*ldu]; };

    for(int64_t j = 0; j < 3*nb; j++) {
        for(int64_t i = 0; i < nb; i++) {
            U(i,j) = 0.0;
        }
    }
    for(int64_t j = 0; j < nb; j++) {
        cblas_dcopy(   j+1, &D(  0, (n-1)*nb + j), 1, &U(  0, j), 1);
        cblas_dcopy(nb-j-1, &D(j+1, (n-1)*nb + j), 1, &L(j+1, j), 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nb, nb, 1.0, l.data(), 2*nb,
            u.data(), nb, 1.0, &B1(nb, nb), 2*nb);
    for(int64_t j = 0; j < nb; j++) {
        cblas_dcopy(j+1,   &D(  0, (n-2)*nb + j), 1, &U(  0,    j), 1);
        cblas_dcopy(nb,    &DU1(0, (n-2)*nb + j), 1, &U(  0, nb+j), 1);
        cblas_dcopy(nb-j-1,&D(j+1, (n-2)*nb + j), 1, &L(j+1,    j), 1);
        cblas_dcopy(nb,    &DL( 0, (n-2)*nb + j), 1, &L( nb,    j), 1);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2*nb, 2*nb, nb, 1.0, l.data(), 2*nb,
            u.data(), nb, 1.0, b1.data(), 2*nb);

    for (int64_t i = nb -1; i >= 0; i--) {
        if(IPIV(i,n-1) != nb+i+1) {
            cblas_dswap(2*nb, &B1(nb+i, 0), 2*nb, &B1(IPIV(i,n-1)-1,0), 2*nb);
        }
    }
    for (int64_t i = nb-1; i >= 0; i--) {
        if (IPIV(i,n-2) != i+1) {
            cblas_dswap(2*nb, &B1(i, 0), 2*nb, &B1(IPIV(i,n-2)-1, 0), 2*nb);
        }
    }
    for(int64_t j = 0; j < nb; j++) {
        for(int64_t i = 0; i < nb; i++) {
            B1(nb+i,    j) = B1(nb+i,    j) - DLCPY(i, (n-2)*nb+j);
            B1(nb+i, nb+j) = B1(nb+i, nb+j) -  DCPY(i, (n-1)*nb+j);
        }
    }
    s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B1(nb, 0), 2*nb);
    double eps = s*s;
    s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B1(nb, nb), 2*nb);
    eps = eps + s*s;

    for (int64_t k = n-3; k >= 0; k--) {
        for(int64_t j = 0; j < nb; j++) {
            cblas_dcopy( j+1,   &D(  0,k*nb + j), 1, &U(0,     j), 1);
            cblas_dcopy(  nb,   &DU1(0,k*nb + j), 1, &U(0,  nb+j), 1);
            cblas_dcopy(  nb,   &DU2(0,k*nb + j), 1, &U(0,2*nb+j), 1);
            cblas_dcopy(nb-j-1, &D(j+1,k*nb + j), 1, &L(j+1,   j), 1);
            cblas_dcopy(  nb,   &DL( 0,k*nb + j), 1, &L(nb,    j), 1);
        }

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 2*nb, 3*nb, nb, 1.0, l.data(), 2*nb, u.data(), nb, 0.0, b.data(), 2*nb);

        for (int64_t j = 0; j < 2*nb; j++) {
            for(int64_t i = 0; i < nb; i++) {
                B(nb+i, nb+j) = B(nb+i, nb+j) + B1(i, j);
            }
        }
        for (int64_t i = nb-1; i >= 0; i--) {
            if (IPIV(i,k) != i+1) {
                cblas_dswap(3*nb, &B(i, 0), 2*nb, &B(IPIV(i,k)-1, 0), 2*nb);
            }
        }
        for(int64_t j = 0; j < nb; j++) {
            for (int64_t i = 0; i < nb; i++) {
                B1(nb+i,     j) = B(nb+i,     j) - DLCPY(i,(k)*nb+j);
                B1(nb+i,  nb+j) = B(nb+i,  nb+j) -  DCPY(i,(k+1)*nb+j);
                B1(nb+i,2*nb+j) = B(nb+i,2*nb+j) -DU1CPY(i,(k+1)*nb+j);
            }
        }
        s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B(0,2*nb), 2*nb);
        eps = eps + s*s;
        s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B1(nb, 0), 2*nb);
        eps = eps + s*s;
        s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B1(nb, nb), 2*nb);
        eps = eps + s*s;
        s   = LAPACKE_dlange(MKL_COL_MAJOR, 'F', nb, nb, &B1(nb, 2*nb), 2*nb);
        eps = eps + s*s;
        for ( int64_t j = 0; j < 2*nb; j++) {
            cblas_dcopy(nb, &B(0, j), 1, &B1(0, j), 1);
        }
    }
    double resid = sqrt(eps)/norm;
   
    return resid;
}

/************************************************************************
* Definition:
* ===========
*   double resid2(int64_t n, int64_t nb, int64_t nrhs, double* dl, double* d, double* du1, double* x, int64_t ldx, double* b, int64_t ldb)

* Purpose:
* ========  
* Given solution X to a system of linear equations AX=B with tridiagonal 
* coefficient matrix A and multiple right hand sides B function RESID2  
* returns max_(i=1,...,NRHS){||AX(i)-B(i)||/||B(i)||}. This quantity 
* provides info on how good the solution is.
* 
* Arguments:
* ==========  
* N (input) int64_t
*     The number of block rows of the matrix A.  N > 0.
*
* NB (input) int64_t
*     The size of blocks.  NB > 0.
*
* NRHS (input) int64_t
*     The number of right hand sides. NRHS >0.
*
* DL (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 subdiagonal blocks (each of size NB by NB) of  
*         the coefficient matrix. The blocks are stored sequentially 
*         block by block.
*
* D (input) double array, dimension (NB) * (N*NB)
*     The array stores N diagonal blocks (each of size NB by NB) 
*         of  the coefficient matrix. The blocks are stored sequentially 
*         block by block.
*
* DU1 (input) double array, dimension (NB) * ((N-1)*NB)
*     The array stores N-1 superdiagonal blocks (each of size NB by NB) 
*         of  the coefficient matrix. The blocks are stored sequentially 
*         block by block.
*
* X (input) double array, dimension (LDX) * (NRHS).
*     The array stores components of the solution to be tested.
*
* LDX (input) int64_t
*     The leading dimension of the array X. LDX >= N*NB.
*
* B (input) double array, dimension (LDB) * (NRHS).
*     The array stores components of the right hand sides.
*
* LDB (input) int64_t
*     The leading dimension of the array B. LDB >= N*NB.
***********************************************************************/
double resid2(int64_t n, int64_t nb, int64_t nrhs, double* dl, double* d, double* du1, double* x, int64_t ldx, double* b, int64_t ldb) {
    
    auto DL  = [=,&dl]  (int64_t i, int64_t j) -> double&  { return  dl[i + j*nb];  };
    auto D   = [=,&d]   (int64_t i, int64_t j) -> double&  { return  d[i + j*nb];   };
    auto DU1 = [=,&du1] (int64_t i, int64_t j) -> double&  { return  du1[i + j*nb]; };
    auto X   = [=,&x]   (int64_t i, int64_t j) -> double&  { return  x[i + j*ldx];  };
    auto B   = [=,&b]   (int64_t i, int64_t j) -> double&  { return  b[i + j*ldb];  };

    // Initializing return value
    std::vector<double> norms(nrhs);

    // Compute norms of RHS vectors
    for (int64_t i = 0; i < nrhs; i++) {
        norms[i] = cblas_dnrm2(nb*n, &B(0,i), 1);
    }

    // Computing B(1)-D(1)*X(1)-DU1(1)*X(2) out of loop     
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, d,   nb,  x,          ldx, 1.0, b, ldb);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, du1, nb,  &X(nb, 0), ldx, 1.0, b, ldb);

    // In the loop computing B(K)-DL(K-1)*X(K-1)-D(K)*X(K)-DU1(K)*X(K+1) 
    for (int64_t k = 1; k < n-1; k++) { 
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0,  &DL(0, (k-1)*nb), nb, &X((k-1)*nb, 0), ldx, 1.0, &B(k*nb, 0), ldb);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0,   &D(0,     k*nb), nb, &X(    k*nb, 0), ldx, 1.0, &B(k*nb, 0), ldb);                      
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &DU1(0,     k*nb), nb, &X((k+1)*nb, 0), ldx, 1.0, &B(k*nb, 0), ldb);
    }

    // Computing B(N)-DL(N-1)*X(N-1)-D(N)*X(N) out of loop     
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0,  &D(0, (n-1)*nb), nb, &X((n-1)*nb, 0), ldx, 1.0, &B((n-1)*nb, 0), ldb);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nb, nrhs, nb, -1.0, &DL(0, (n-2)*nb), nb, &X((n-2)*nb, 0), ldx, 1.0, &B((n-1)*nb, 0), ldb);

    // Compute norms of residual vectors divided by norms of RHS vectors
    double res = 0.0;
    for (int64_t i = 0; i < nrhs; i++) {
        double s  = cblas_dnrm2(n*nb, &B(0,i), 1);
        res = std::max<double>(res, s/norms[i]);
    }

    return res;
}
