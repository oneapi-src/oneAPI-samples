//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Snippet begin
#include "mkl.h"
#include "mkl_omp_offload.h"
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#if defined(PRECISION)
#if PRECISION == 1
#define FLOAT double
#else
#define FLOAT float
#endif
#else
#define PRECISION 1
#define FLOAT double
#endif

#define index(i, j, ld) (((j) * (ld)) + (i))

#define RAND() ((FLOAT)rand() / (FLOAT)RAND_MAX * 2.0 - 1.0)

#define MALLOC(x) mkl_malloc((x), 64);
#define FREE mkl_free

#define MALLOC_CHECK(p)                                                        \
  if (p == NULL) {                                                             \
    fprintf(stderr, "%s:%d: memory allocation error\n", __FILE__, __LINE__);   \
    return EXIT_FAILURE;                                                       \
  }

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

void printMat(FLOAT *P, int m, int n, int ld) {
  int i, j;
  for (i = 0; i < m; i++) {
    printf("\n");
    for (j = 0; j < n; j++)
      printf("%f ", P[index(i, j, ld)]);
  }
  printf("\n");
}

void gemm_naive(int m, int n, int k, FLOAT alpha, const FLOAT *A, int lda,
                const FLOAT *B, int ldb, FLOAT beta, FLOAT *C, int ldc) {
  int i, j, kk;
  FLOAT temp;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      temp = 0.0;
      for (kk = 0; kk < k; kk++) {
        temp += alpha * A[index(i, kk, lda)] * B[index(kk, j, ldb)];
      }
      C[index(i, j, ldc)] = temp + beta * C[index(i, j, ldc)];
    }
  }
}

FLOAT infinity_norm_error(int m, int n, int ld, const FLOAT *ans,
                          const FLOAT *ref) {
  int i, j, ind;
  FLOAT norm = 0.0, temp;
  for (i = 0; i < m; i++) {
    temp = 0.0;
    for (j = 0; j < n; j++) {
      ind = index(i, j, ld);
      temp += fabs(ref[ind] - ans[ind]);
    }
    norm = max(norm, temp);
  }
  return norm;
}

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i;
  i = gettimeofday(&tp, &tzp);
  if (i != 0) {
    fprintf(stderr, "%s:%d: timing error %d\n", __FILE__, __LINE__, i);
    return EXIT_FAILURE;
  }
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#if defined(USE_MKL)
int dnum = 0;
#endif

#if PRECISION == 1
#define LD_ALIGN 256
#define LD_BIAS 8
#else
#define LD_ALIGN 512
#define LD_BIAS 16
#endif

#define HPL_PTR(ptr_, al_) ((((size_t)(ptr_) + (al_)-1) / (al_)) * (al_))

#if defined(PAD_LD)
static inline int getld(int x) {
  int ld;
  ld = HPL_PTR(x, LD_ALIGN); // Rule 1
  if (ld - LD_BIAS >= x)
    ld -= LD_BIAS;
  else
    ld += LD_BIAS; // Rule 2
  return ld;
}
#else
static inline int getld(int x) { return x; }
#endif

int main(int argc, char **argv) {
  int i, j;

  if ((argc < 4) || (argc > 4 && argc < 8)) {
    printf("Performs a DGEMM test C = alpha*A*B + beta*C\n");
    printf("A matrix is MxK and B matrix is KxN\n");
    printf("All matrices are stored in column-major format\n");
    printf("Run as ./dgemm_cublas <M> <K> <N> [<alpha> <beta> <iterations> "
           "<verify>]\n");
    printf("Required inputs are:\n");
    printf("      M: number of rows of matrix A\n");
    printf("      K: number of cols of matrix A\n");
    printf("      N: number of cols of matrix B\n");
    printf("Optional inputs are (all must be provided if providing any):\n");
    printf("      alpha: scalar multiplier (default: 1.0)\n");
    printf("      beta:  scalar multiplier (default: 0.0)\n");
    printf("      iterations: number of blocking DGEMM calls to perform "
           "(default: 10)\n");
    printf("      verify: set to 1 to check solution against CPU reference, "
           "not recommended for large M|K|N (default: 0)\n");
    return EXIT_FAILURE;
  }

  FLOAT alpha, beta;
  int niter, verify;
  int HA = atoi(argv[1]);
  int WA = atoi(argv[2]);
  int WB = atoi(argv[3]);

  if ((HA == 0) || (WA == 0) || (WB == 0))
    exit(1);

  if (argc > 4) {

#if PRECISION == 1
    sscanf(argv[4], "%lf", &alpha);
    sscanf(argv[5], "%lf", &beta);
#else
    sscanf(argv[4], "%f", &alpha);
    sscanf(argv[5], "%f", &beta);
#endif
    niter = atoi(argv[6]);
    verify = atoi(argv[7]);
  } else {
    alpha = 1.0;
    beta = 0.0;
    niter = 10;
    verify = 0;
  }

#if PRECISION == 1
  printf("DGEMM performance test\n");
#else
  printf("SGEMM performance test\n");
#endif

  int HB = WA;
  int WC = WB;
  int HC = HA;

  int ldA = getld(HA);
  int ldB = getld(HB);
  int ldC = getld(HC);

  printf("M = %d, K = %d, N = %d, ldA = %d, ldB = %d, ldC = %d, alpha = %f, "
         "beta = %f, iterations = %d, verify? = %d\n",
         HA, WA, WB, ldA, ldB, ldC, alpha, beta, niter, verify);

  double start_t, end_t, tot_t = 0.0, best_t = DBL_MAX;

  /*ALLOCATE HOST ARRAYS*/
  FLOAT *A = (FLOAT *)MALLOC(ldA * WA * sizeof(FLOAT));
  MALLOC_CHECK(A);
  FLOAT *B = (FLOAT *)MALLOC(ldB * WB * sizeof(FLOAT));
  MALLOC_CHECK(B);
  FLOAT *C = (FLOAT *)MALLOC(ldC * WC * sizeof(FLOAT));
  MALLOC_CHECK(C);
  FLOAT *Cref = NULL;
  if (verify) {
    Cref = (FLOAT *)MALLOC(ldC * WC * sizeof(FLOAT));
    MALLOC_CHECK(Cref);
  }

  printf("\n---------------------\n");
  printf("Array A: %d x %d\n", ldA, WA);
  printf("Array B: %d x %d\n", ldB, WB);
  printf("Array C: %d x %d\n", ldC, WC);
  printf("---------------------\n");

  /*INITIALIZE WITH PSEUDO-RANDOM DATA*/
  srand(2864);
  for (j = 0; j < WA; j++)
    for (i = 0; i < HA; i++)
      A[index(i, j, ldA)] = RAND();

  for (j = 0; j < WB; j++)
    for (i = 0; i < HB; i++)
      B[index(i, j, ldB)] = RAND();

  if (beta != 0.0) {
    for (j = 0; j < WC; j++)
      for (i = 0; i < HC; i++)
        C[index(i, j, ldC)] = RAND();
  } else {
    for (j = 0; j < WC; j++)
      for (i = 0; i < HC; i++)
        C[index(i, j, ldC)] = 0.0;
  }

  if (verify) {
    for (j = 0; j < WC; j++)
      for (i = 0; i < HC; i++)
        Cref[index(i, j, ldC)] = C[index(i, j, ldC)];
  }

#if defined(USE_MKL)
  size_t sizea = (size_t)ldA * WA;
  size_t sizeb = (size_t)ldB * WB;
  size_t sizec = (size_t)ldC * WC;

#pragma omp target data map(to: A [0:sizea], B [0:sizeb]) map(tofrom: C [0:sizec])
  {

    // warm-up run
    #pragma omp dispatch
#if PRECISION == 1
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HA, WB, WA, alpha, A,
                ldA, B, ldB, beta, C, ldC);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HA, WB, WA, alpha, A,
                ldA, B, ldB, beta, C, ldC);
#endif

    // run gemm on gpu, using dispatch construct
    for (i = 0; i < niter; i++) {
      start_t = mysecond();

      #pragma omp dispatch
#if PRECISION == 1
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HA, WB, WA, alpha,
                  A, ldA, B, ldB, beta, C, ldC);
#else
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, HA, WB, WA, alpha,
                  A, ldA, B, ldB, beta, C, ldC);
#endif

      end_t = mysecond();
      tot_t += end_t - start_t;
      best_t = min(best_t, end_t - start_t);
    } // end for
  }
#endif // end #pragma omp target data

  double tflop_count;
  tflop_count = (double)2.0 * HA * WB * WA;
  if (beta != 0.0)
    tflop_count += (double)HA * WB;
  tflop_count *= 1.E-12;

  printf("Total runtime for %d iterations: %f seconds.\n", niter, tot_t);
  printf("Mean TFLOP/s: %f\n", (double)niter * tflop_count / tot_t);
  printf("Best TFLOP/s: %f\n", (double)tflop_count / best_t);

  if (verify) {
    // compute reference solution on host (1 added to niter to account for the
    // warm-up run)
    for (i = 0; i < (niter + 1); i++) {
      gemm_naive(HA, WB, WA, alpha, A, ldA, B, ldB, beta, Cref, ldC);
    }

    printf("Error Matrix Infinity Norm = %f\n",
           infinity_norm_error(HA, WB, ldC, C, Cref));
  }

  FREE(A);
  FREE(B);
  FREE(C);
  if (verify)
    FREE(Cref);

  return EXIT_SUCCESS;
}
// Snippet end
