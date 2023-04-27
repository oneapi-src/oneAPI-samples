//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mkl.h"
#include "mkl_omp_offload.h"

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define EPSILON 0.0001

int main()
{
    double *A, *B, *C, *C_fl;
    int64_t m, n, k;
    double alpha, beta;
    double sum;
    int64_t i, j, q;
    int fail;

    printf ("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
            " Intel oneMKL function dgemm, where A, B, and  C are matrices and \n"
            " alpha and beta are double precision scalars\n\n");

    m = 2000, k = 200, n = 1000;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%li x %li) and matrix B(%li x %li)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (double *)mkl_malloc( m * k * sizeof( double ), 64 );
    B = (double *)mkl_malloc( k * n * sizeof( double ), 64 );
    C = (double *)mkl_malloc( m * n * sizeof( double ), 64 );
    C_fl = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

    if (A == NULL || B == NULL || C == NULL || C_fl == NULL) {
      printf( "\n ERROR: Cannot allocate memory for matrices. Exiting... \n\n");
      return 1;
    }

    printf (" Intializing matrices \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
        C_fl[i] = 0.0;
    }

    printf (" Computing matrix product using Intel oneMKL dgemm function via CBLAS interface \n\n");

    #pragma omp target data map(to: A[0:m*k], B[0:k*n]) map(tofrom: C[0:m*n])
    {
       #pragma omp target variant dispatch use_device_ptr(A, B, C)
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   m, n, k, alpha, A, k, B, n, beta, C, n);
    }

    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(n,6); j++) {
            printf ("%12.5G", C[j+i*n]);
        }
        printf ("\n");
    }

    printf (" Computing matrix product using for-loops \n");

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (q = 0; q < k; q++) {
                sum += A[k*i+q] * B[n*q+j];
            }
            C_fl[n*i+j] = alpha * sum + beta * C_fl[n*i+j];
        }
    }

    printf ("\n Top left corner of matrix C_fl: \n");
    for (i=0; i<min(m,6); i++) {
        for (j=0; j<min(n,6); j++) {
            printf ("%12.5G", C_fl[j+i*n]);
        }
        printf ("\n");
    }

    printf (" Computations completed. Verifying... \n\n");

    fail = 0;
    for (i = 0; i < (m*n); i++) {
        if (fabs(C[i] - C_fl[i]) > EPSILON) {
            fail = 1;
            break;
	}
    }

    if (fail)
        printf ("\n **** FAIL **** \n");
    else
        printf ("\n **** PASS **** \n");

    printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return fail;
}
// Snippet end
