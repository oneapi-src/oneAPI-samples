//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <stdio.h>
#include "mkl.h"              //# main mkl header
#include "mkl_omp_offload.h"  //# mkl OMP Offload interface

int dnum = 0;

int main() {

    //# dimensions
    MKL_INT m = 3, n = 3, k = 3;
    //# leading dimensions
    MKL_INT ldA = k, ldB = n, ldC = n;
    //# scalar multipliers
    double alpha = 1.0;
    double beta = 1.0;
    //# matrix data
    double *A = (double *)malloc(m * k * sizeof(double));
    double *B = (double *)malloc(k * n * sizeof(double));
    double *C = (double *)malloc(m * n * sizeof(double));

    //# define matrix A as the 3x3 matrix
    //# {{ 1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i*m+j] = (double)(i*m+j) + 1.0;
        }
    }

    //# define matrix B as the identity matrix
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) B[i*k+j] = 1.0;
            else B[i*k+j] = 0.0;
        }
    }

    //# initialize C as a 0 matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*m+j] = 0.0;
        }
    }

    MKL_INT sizeA = m*k;
    MKL_INT sizeB = k*n;
    MKL_INT sizeC = m*n;
    
    //# Below are the two compiler directives necessary to offload the GEMM operation
    //# we are using 'dgemm' to specify we are using double-precision values
    
    //# The outer directive maps input data (matrices A & B) 'to' the device.
    //# It also maps output data (matrix C) 'from' the device so that the results of the operation are returned.
    //# Finally, this directive specifies device number 0, which should interact with an available GPU.
    
    //# The inner directive dispatches the correct version of the contained operation, again specifying the device number.
    //# This directive also uses the 'use_devce_ptr' statement to specify the data we are working with (in this case, arrays A, B, & C).
    
    //# Uncomment the two 'pragma' lines below. (Do not remove the '#' character)
    
    //#pragma omp target data map(to:A[0:sizeA],B[0:sizeB]) map(from:C[0:sizeC]) device(dnum)
    {
        //#pragma omp target variant dispatch device(dnum) use_device_ptr(A, B, C)
        {
            dgemm("N", "N", &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC);
        }
    }

    int status = 0;

    //# verify C matrix
    printf("\n");
    printf("C = \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (A[i*m+j] != C[i*m+j]) status = 1;
            printf("%0.0f ", C[i*m+j]);
        }
        printf("\n");
    }
    printf("\n");

    //# free matrix data
    free(A);
    free(B);
    free(C);

    status == 0 ? printf("Verified: A = C\n") : printf("Failed: A != C\n");

    return status;
}