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
#define epsilon 0.0000001f

bool compare(double x, double y)
{
    // returns true if x and y are the same
    return fabs(x - y) <= epsilon;
}

int main()
{
    double *A1, *B1, *C1, *C1_fl;
    double *A2, *B2, *C2, *C2_fl;
    int m, n, k, i, j, q;
    double alpha, beta;
    double sum;
    int fail;
    double t_start, t_end;

    m = 2000, k = 200, n = 1000;
    alpha = 1.0; beta = 0.0;

    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A1 = (double *)mkl_malloc (m*k*sizeof( double ), 64 );
    B1 = (double *)mkl_malloc (k*n*sizeof( double ), 64 );
    C1 = (double *)mkl_malloc (m*n*sizeof( double ), 64 );
    C1_fl = (double *)mkl_malloc (m*n*sizeof( double ), 64 );

    A2 = (double *)mkl_malloc (m*k*sizeof( double ), 64 );
    B2 = (double *)mkl_malloc (k*n*sizeof( double ), 64 );
    C2 = (double *)mkl_malloc (m*n*sizeof( double ), 64 );
    C2_fl = (double *)mkl_malloc (m*n*sizeof( double ), 64 );

    if (A1 == NULL || B1 == NULL || C1 == NULL || C1_fl == NULL ||
        A2 == NULL || B2 == NULL || C2 == NULL || C2_fl == NULL) {
      printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
      return 1;
    }

    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A1[i] = A2[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B1[i] = B2[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C1[i]    = C2[i]    = 0.0;
        C1_fl[i] = C2_fl[i] = 0.0;
    }

    printf (" \nComputing matrix product using Intel MKL cblas_dgemm function \n");

    t_start = omp_get_wtime();

    #pragma omp target data				   \
      map(to: A1[0:m*k], B1[0:k*n], A2[0:m*k], B2[0:k*n])  \
      map(tofrom: C1[0:m*n], C2[0:m*n])
    {
       #pragma omp dispatch
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   m, n, k, alpha, A1, k, B1, n, beta, C1, n);

       #pragma omp dispatch
       cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   m, n, k, alpha, A2, k, B2, n, beta, C2, n);
    }

    t_end = omp_get_wtime();

    printf ("\n Top left corner of matrix C1: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C1[j+i*n]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix C2: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C2[j+i*n]);
      }
      printf ("\n");
    }

    printf (" \nComputing matrix product using for-loops \n");

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (q = 0; q < k; q++)
                sum += A1[k*i+q] * B1[n*q+j];
            C1_fl[n*i+j] = sum;
        }
    }

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (q = 0; q < k; q++)
                sum += A2[k*i+q] * B2[n*q+j];
            C2_fl[n*i+j] = sum;
        }
    }

    printf ("\n Top left corner of matrix C1: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C1_fl[j+i*n]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix C2: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C2_fl[j+i*n]);
      }
      printf ("\n");
    }

    printf ("\n Computations completed. Verifying... \n\n");

    fail = 0;
    for (i = 0; i < (m*n); i++) {
      if (! compare(C1[i], C1_fl[i]) || ! compare(C2[i], C2_fl[i])) {
          fail = 1;
          break;
      }
    }

    if (fail) {
        printf (" **** FAIL **** \n");
    }
    else {
        printf(" time = %lf seconds\n", t_end - t_start);
        printf (" **** PASS **** \n");
    }

    mkl_free(A1);
    mkl_free(B1);
    mkl_free(C1);
    mkl_free(C1_fl);
    mkl_free(A2);
    mkl_free(B2);
    mkl_free(C2);
    mkl_free(C2_fl);

    return 0;
}
// Snippet end
