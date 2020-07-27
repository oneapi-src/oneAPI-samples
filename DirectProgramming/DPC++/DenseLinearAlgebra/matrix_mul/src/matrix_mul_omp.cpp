//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <float.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <limits>

using namespace std;

// Matrix size constants.
constexpr int m_size = 150 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */
float a[M][N];
float b[N][P];
float c[M][P];

/**
 * Perform matrix multiplication on CPU with OpenMP.
 */
void MatrixMulOpenMpCpu(float (*a)[N], float (*b)[P], float (*c)[P]);

/**
 * Perform matrix multiplication on GPU with OpenMP offloading.
 */
void __attribute__((noinline)) MatrixMulOpenMpGpuOffloading();

/**
 * Perform matrix multiplication on host to verify results from OpenMP.
 */
int VerifyResult(float (*c_back)[P]);

int main(void) {
  int Result1, Result2;

  cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
       << ") * b(" << N << "," << P << ")\n";

  cout << "Running on " << omp_get_num_devices() << " device(s)\n";
  cout << "The default device id: " << omp_get_default_device() << "\n";

  MatrixMulOpenMpCpu(a, b, c);
  cout << "Result of matrix multiplication using OpenMP: ";
  Result1 = VerifyResult(c);

  MatrixMulOpenMpGpuOffloading();
  cout << "Result of matrix multiplication using GPU offloading: ";
  Result2 = VerifyResult(c);

  return Result1 || Result2;
}

void MatrixMulOpenMpCpu(float (*a)[N], float (*b)[P], float (*c)[P]) {
  int i, j, k;

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a[i][j] = 1.0f;

  // Each column of b is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b[i][j] = i + 1.0f;

  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c[i][j] = 0.0f;

// Parallelize by row. The threads don't need to synchronize at
// loop end, so "nowait" can be used.
#pragma omp for nowait private(i, j, k)
  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      // Each element of the product is just the sum 1+2+...+n
      for (j = 0; j < P; j++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void __attribute__((noinline)) MatrixMulOpenMpGpuOffloading() {
  int i, j, k;

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a[i][j] = 1.0f;

  // Each column of b is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b[i][j] = i + 1.0f;

  // c is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c[i][j] = 0.0f;

// Parallelize on target device.
#pragma omp target teams distribute parallel for map(to : a, b) \
  map(tofrom : c) thread_limit(128)
  {
    for (i = 0; i < M; i++) {
      for (k = 0; k < N; k++) {
        // Each element of the product is just the sum 1+2+...+n
        for (j = 0; j < P; j++) {
          c[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }
}

bool ValueSame(float a, float b) {
  return fabs(a - b) < numeric_limits<float>::epsilon();
}

int VerifyResult(float (*c_back)[P]) {
  // Check that the results are correct by comparing with host computing.
  int i, j, k;

  // 2D arrays on host side.
  float(*a_host)[N] = new float[M][N];
  float(*b_host)[P] = new float[N][P];
  float(*c_host)[P] = new float[M][P];

  // Each element of matrix a is 1.
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) a_host[i][j] = 1.0f;

  // Each column of b_host is the sequence 1,2,...,N
  for (i = 0; i < N; i++)
    for (j = 0; j < P; j++) b_host[i][j] = i + 1.0f;

  // c_host is initialized to zero.
  for (i = 0; i < M; i++)
    for (j = 0; j < P; j++) c_host[i][j] = 0.0f;

  for (i = 0; i < M; i++) {
    for (k = 0; k < N; k++) {
      // Each element of the product is just the sum 1+2+...+n
      for (j = 0; j < P; j++) {
        c_host[i][j] += a_host[i][k] * b_host[k][j];
      }
    }
  }

  bool mismatch_found = false;

  // Compare host side results with the result buffer from device side: print
  // mismatched data 5 times only.
  int print_count = 0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < P; j++) {
      if (!ValueSame(c_back[i][j], c_host[i][j])) {
        cout << "Fail - The result is incorrect for element: [" << i << ", "
             << j << "], expected: " << c_host[i][j]
             << ", but found: " << c_back[i][j] << "\n";
        mismatch_found = true;
        print_count++;
        if (print_count == 5) break;
      }
    }

    if (print_count == 5) break;
  }

  delete[] a_host;
  delete[] b_host;
  delete[] c_host;

  if (!mismatch_found) {
    cout << "Success - The results are correct!\n";
    return 0;
  } else {
    cout << "Fail - The results mismatch!\n";
    return -1;
  }
}
