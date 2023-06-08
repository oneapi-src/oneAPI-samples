//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
/*******************************************************************************
* Copyright 2019-2021 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

// Snippet begin
#include <stdio.h>
#include <omp.h>
#include "mkl.h"
#include "mkl_omp_offload.h"
#include "common.h"

#define GROUP_COUNT 3

int dnum = 0;

int main() {

    CBLAS_LAYOUT layout = (rand_int(0,1) == 0) ? CblasColMajor : CblasRowMajor;
    CBLAS_TRANSPOSE *transA, *transB;
    MKL_INT *m, *n, *k, *lda, *ldb, *ldc;
    double *alpha, *beta;
    MKL_INT *group_size, *sizea_array, *sizeb_array, *sizec_array, total_batch_size = 0, sizea, sizeb, sizec;
    double **a_array, **b_array, **c_array, **c_ref_array;
    double **a_array_dev, **b_array_dev, **c_array_dev;

    transA = (CBLAS_TRANSPOSE *)mkl_malloc(GROUP_COUNT * sizeof(CBLAS_TRANSPOSE), 64);
    transB = (CBLAS_TRANSPOSE *)mkl_malloc(GROUP_COUNT * sizeof(CBLAS_TRANSPOSE), 64);

    m = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    n = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    k = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    lda = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    ldb = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    ldc = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    group_size = (MKL_INT *)mkl_malloc(GROUP_COUNT * sizeof(MKL_INT), 64);
    alpha = (double *)mkl_malloc(GROUP_COUNT * sizeof(double), 64);
    beta = (double *)mkl_malloc(GROUP_COUNT * sizeof(double), 64);

    if ((m == NULL) || (n == NULL) || (k == NULL) || (lda == NULL) || (ldb == NULL) || (ldc == NULL) ||
        (group_size == NULL) || (alpha == NULL) || (beta == NULL)) {
        printf("Cannot allocate input arrays\n");
        return 1;
    }

    MKL_INT i, j, p, idx;

    for (i = 0; i < GROUP_COUNT; i++) {
        transA[i] = (rand_int(0,1) == 0) ? CblasNoTrans : CblasTrans;
        transB[i] = (rand_int(0,1) == 0) ? CblasNoTrans : CblasTrans;
        alpha[i] = rand_double_scalar();
        beta[i] = rand_double_scalar();
        m[i] = rand_int(1, 20);
        n[i] = rand_int(1, 20);
        k[i] = rand_int(1, 20);
        lda[i] = MAX(m[i], k[i]);
        ldb[i] = MAX(k[i], n[i]);
        ldc[i] = MAX(m[i], n[i]);
        group_size[i] = rand_int(1, 10);
        total_batch_size += group_size[i];
#ifdef MKL_ILP64
        printf("Group %lld: layout = %s, transA = %s, transB = %s, m = %lld, n = %lld, k = %lld, lda = %lld, ldb = %lld, ldc = %lld, alpha = %lf, beta = %lf, group_size = %lld\n",
               i, (layout == CblasColMajor) ? "Column Major" : "Row Major",
               (transA[i] == CblasNoTrans) ? "Non Transpose" : "Transpose",
               (transB[i] == CblasNoTrans) ? "Non Transpose" : "Transpose",
               m[i], n[i], k[i], lda[i], ldb[i], ldc[i], alpha[i], beta[i],  group_size[i]);
#else
        printf("Group %d: layout = %s, transA = %s, transB = %s, m = %d, n = %d, k = %d, lda = %d, ldb = %d, ldc = %d, alpha = %lf, beta = %lf, group_size = %d\n",
               i, (layout == CblasColMajor) ? "Column Major" : "Row Major",
               (transA[i] == CblasNoTrans) ? "Non Transpose" : "Transpose",
               (transB[i] == CblasNoTrans) ? "Non Transpose" : "Transpose",
               m[i], n[i], k[i], lda[i], ldb[i], ldc[i], alpha[i], beta[i],  group_size[i]);
#endif
    }

    sizea_array = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * total_batch_size, 64);
    sizeb_array = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * total_batch_size, 64);
    sizec_array = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * total_batch_size, 64);

    a_array = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    b_array = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    c_array = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    a_array_dev = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    b_array_dev = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    c_array_dev = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);
    c_ref_array = (double **)mkl_malloc(sizeof(double *) * total_batch_size, 64);

    if ((sizea_array == NULL) || (sizeb_array == NULL) || (sizec_array == NULL) || (a_array == NULL) ||
        (b_array == NULL) || (c_array == NULL) || (a_array_dev == NULL) || (b_array_dev == NULL) ||
        (c_array_dev == NULL) || (c_ref_array == NULL)) {
        printf("Cannot allocate matrices and size arrays\n");
        return 1;
    }

    idx = 0;
    for (i = 0; i < GROUP_COUNT; i++) {
        sizea = (((layout == CblasRowMajor) && (transA[i] == CblasTrans)) ||
                 ((layout == CblasColMajor) && (transA[i] == CblasNoTrans))) ? lda[i] * k[i] : m[i] * lda[i];
        sizeb = (((layout == CblasRowMajor) && (transB[i] == CblasTrans)) ||
                 ((layout == CblasColMajor) && (transB[i] == CblasNoTrans))) ? ldb[i] * n[i] : k[i] * ldb[i];
        sizec = (layout == CblasColMajor) ? ldc[i] * n[i] : ldc[i] * m[i];
        for (j = 0; j < group_size[i]; j++) {
            a_array[idx] = (double *)mkl_malloc(sizeof(double) * sizea, 64);
            a_array_dev[idx] = a_array[idx];
            sizea_array[idx] = sizea;
            if (a_array[idx] == NULL) {
                printf("cannot allocate a matrices\n");
                return 1;
            }
            b_array[idx] = (double *)mkl_malloc(sizeof(double) * sizeb, 64);
            b_array_dev[idx] = b_array[idx];
            sizeb_array[idx] = sizeb;
            if (b_array[idx] == NULL) {
                printf("cannot allocate b matrices\n");
                return 1;
            }
            c_array[idx] = (double *)mkl_malloc(sizeof(double) * sizec, 64);
            c_array_dev[idx] = c_array[idx];
            sizec_array[idx] = sizec;
            if (c_array[idx] == NULL) {
                printf("cannot allocate c matrices\n");
                return 1;
            }
            c_ref_array[idx] = (double *)mkl_malloc(sizeof(double) * sizec, 64);
            if (c_ref_array[idx] == NULL) {
                printf("cannot allocate c_ref matrices\n");
                return 1;
            }
            init_double_array(sizea, a_array[idx], 1);
            init_double_array(sizeb, b_array[idx], 1);
            init_double_array(sizec, c_array[idx], 1);
            for (p = 0; p < sizec_array[idx]; p++) c_ref_array[idx][p] = c_array[idx][p];
            idx++;
        }
    }

    // run gemm_batch on host, use standard oneMKL interface
    cblas_dgemm_batch(layout, transA, transB, m, n, k, alpha, (const double **) a_array, lda,
                      (const double **) b_array, ldb, beta, c_ref_array, ldc, GROUP_COUNT, group_size);

    double *a, *b, *c;
    for (i = 0; i < total_batch_size; i++) {
        a = a_array[i];
        b = b_array[i];
        c = c_array[i];
#pragma omp target enter data map(to:a[0:sizea_array[i]],b[0:sizeb_array[i]],c[0:sizec_array[i]])
#pragma omp target data use_device_ptr(a,b,c)
        {
            a_array_dev[i] = a;
            b_array_dev[i] = b;
            c_array_dev[i] = c;
        }
    }
#pragma omp target data map(to:a_array_dev[0:total_batch_size], \
                            b_array_dev[0:total_batch_size], \
                            c_array_dev[0:total_batch_size]) device(dnum)
    {

#pragma omp dispatch
        cblas_dgemm_batch(layout, transA, transB, m, n, k, alpha, (const double **) a_array_dev, lda, (const double **) b_array_dev, ldb, beta, c_array_dev, ldc, GROUP_COUNT, group_size);

    }

    for (i = 0; i < total_batch_size; i++) {
        a = a_array[i];
        b = b_array[i];
        c = c_array[i];
#pragma omp target exit data map(from:a[0:sizea_array[i]],b[0:sizeb_array[i]],c[0:sizec_array[i]])
    }

    double computed, reference, diff;
    MKL_INT l;
    idx = 0;
    for (p = 0; p < GROUP_COUNT; p++) {
        for (l = 0; l < group_size[p]; l++) {
            for (i = 0; i < m[p]; i++) {
                for (j = 0; j < n[p]; j++) {
                    if (layout == CblasColMajor) {
                        computed = c_array[idx][i + j * ldc[p]];
                        reference = c_ref_array[idx][i + j * ldc[p]];
                    }
                    else {
                        computed = c_array[idx][j + i * ldc[p]];
                        reference = c_ref_array[idx][j + i * ldc[p]];
                    }
                    diff = computed - reference;
                    diff = (diff > 0) ? diff : -diff;
                    if (diff > 0.0001) {
#ifdef MKL_ILP64
                        printf("Error in matrix %lld (group = %lld, matrix index in group = %lld) at index [%lld][%lld], computed = %lf, reference = %lf, difference = %lf\n", idx, p, l, i, j, computed, reference, diff);
#else
                        printf("Error in matrix %d at index [%d][%d], computed = %lf, reference = %lf, difference = %lf\n", idx, i, j, computed, reference, diff);
#endif

                       free_double_matrices(a_array, total_batch_size);
                       free_double_matrices(b_array, total_batch_size);
                       free_double_matrices(c_array, total_batch_size);
                       free_double_matrices(c_ref_array, total_batch_size);
                       mkl_free(a_array);
                       mkl_free(b_array);
                       mkl_free(c_array);
                       mkl_free(c_ref_array);
                       mkl_free(a_array_dev);
                       mkl_free(b_array_dev);
                       mkl_free(c_array_dev);
                       mkl_free(sizea_array);
                       mkl_free(sizeb_array);
                       mkl_free(sizec_array);
                       mkl_free(transA); mkl_free(transB);
                       mkl_free(m); mkl_free(n); mkl_free(k);
                       mkl_free(lda); mkl_free(ldb); mkl_free(ldc); mkl_free(group_size);
                       mkl_free(alpha); mkl_free(beta);
                       return 1;
                    }
                }
            }
            idx++;
        }
    }

    printf("Validation PASSED\n");
    free_double_matrices(a_array, total_batch_size);
    free_double_matrices(b_array, total_batch_size);
    free_double_matrices(c_array, total_batch_size);
    free_double_matrices(c_ref_array, total_batch_size);
    mkl_free(a_array);
    mkl_free(b_array);
    mkl_free(c_array);
    mkl_free(c_ref_array);
    mkl_free(a_array_dev);
    mkl_free(b_array_dev);
    mkl_free(c_array_dev);
    mkl_free(sizea_array);
    mkl_free(sizeb_array);
    mkl_free(sizec_array);
    mkl_free(transA); mkl_free(transB);
    mkl_free(m); mkl_free(n); mkl_free(k);
    mkl_free(lda); mkl_free(ldb); mkl_free(ldc); mkl_free(group_size);
    mkl_free(alpha); mkl_free(beta);
    return 0;
}
// Snippet end
