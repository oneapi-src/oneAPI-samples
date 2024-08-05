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
    // return true is x and y are the same
    return (fabs(x - y) <= epsilon);
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

    printf (" \nComputing matrix product using Intel MKL cblas_dgemm_batch function \n");

    #define GRP_COUNT 1  // 1 group

    MKL_INT group_count = GRP_COUNT;
    MKL_INT group_sizes[GRP_COUNT] = {2};  // 8 matrix multiplications

    CBLAS_TRANSPOSE transa_array[GRP_COUNT] = {CblasNoTrans};
    CBLAS_TRANSPOSE transb_array[GRP_COUNT] = {CblasNoTrans};

    MKL_INT m_array[GRP_COUNT] = {m};
    MKL_INT n_array[GRP_COUNT] = {n};
    MKL_INT k_array[GRP_COUNT] = {k};

    MKL_INT lda_array[GRP_COUNT] = {k};
    MKL_INT ldb_array[GRP_COUNT] = {n};
    MKL_INT ldc_array[GRP_COUNT] = {n};

    double alpha_array[GRP_COUNT] = {alpha};
    double beta_array[GRP_COUNT]  = {beta};

    // Number of matrix multiplications = 2
    double **a_array, **b_array, **c_array;
    a_array = (double **)mkl_calloc(2, sizeof( double* ), 64);
    b_array = (double **)mkl_calloc(2, sizeof( double* ), 64);
    c_array = (double **)mkl_calloc(2, sizeof( double* ), 64);

    t_start = omp_get_wtime();

    // Call cblas_dgemm_batch
    #pragma omp target enter data \
      map(to: A1[0:m*k], B1[0:k*n], C1[0:m*n]) \
      map(to: A2[0:m*k], B2[0:k*n], C2[0:m*n])

    #pragma omp target data use_device_ptr(A1, B1, C1, A2, B2, C2)
    {
      a_array[0] = A1, a_array[1] = A2;
      b_array[0] = B1, b_array[1] = B2;
      c_array[0] = C1, c_array[1] = C2;
    }

    #pragma omp target data				   \
      map(to:a_array[0:2], b_array[0:2], c_array[0:2])
    {
       #pragma omp dispatch
         cblas_dgemm_batch (
             CblasRowMajor,
             transa_array,
             transb_array,
             m_array,
             n_array,
             k_array,
             alpha_array,
             (const double **)a_array,
             lda_array,
             (const double **)b_array,
             ldb_array,
             beta_array,
             c_array,
             ldc_array,
             group_count,
             group_sizes);
    } // end target data map

    #pragma omp target exit data \
      map(from: C1[0:m*n], C2[0:m*n])

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
