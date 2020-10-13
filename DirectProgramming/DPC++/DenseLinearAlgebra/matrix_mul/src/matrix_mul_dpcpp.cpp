//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

/**
 * Matrix_mul multiplies two large matrices both the CPU and the offload device,
 * then compares results. If the code executes on both CPU and the offload
 * device, the name of the offload device and a success message are displayed.
 *
 * For comprehensive instructions regarding DPC++ Programming, go to
 * https://software.intel.com/en-us/oneapi-programming-guide and search based on
 * relevant terms noted in the comments.
 */

#include <CL/sycl.hpp>
#include <iostream>
#include <limits>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace sycl;

/**
 * Each element of the product matrix c[i][j] is computed from a unique row and
 * column of the factor matrices, a[i][k] and b[k][j]
 */

// Matrix size constants.
constexpr int m_size = 150 * 8;  // Must be a multiple of 8.
constexpr int M = m_size / 8;
constexpr int N = m_size / 4;
constexpr int P = m_size / 2;

/**
 * Perform matrix multiplication on host to verify results from device.
 */
int VerifyResult(float (*c_back)[P]);

int main() {
  // Host memory buffer that device will write data back before destruction.
  float(*c_back)[P] = new float[M][P];

  // Intialize c_back
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  try {
    queue q(default_selector{}, dpc_common::exception_handler);

    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back

    buffer<float, 2> a_buf(range(M, N));
    buffer<float, 2> b_buf(range(N, P));
    buffer c_buf(reinterpret_cast<float *>(c_back), range(M, P));

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    // Using three command groups to illustrate execution order. The use of
    // first two command groups for initializing matrices is not the most
    // efficient way. It just demonstrates the implicit multiple command group
    // execution ordering.

    // Submit command group to queue to initialize matrix a
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device.
      accessor a(a_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(M, N), [=](auto index) {
        // Each element of matrix a is 1.
        a[index] = 1.0f;
      });
    });

    // Submit command group to queue to initialize matrix b
    q.submit([&](auto &h) {
      // Get write only access to the buffer on a device
      accessor b(b_buf, h, write_only);

      // Execute kernel.
      h.parallel_for(range(N, P), [=](auto index) {
        // Each column of b is the sequence 1,2,...,N
        b[index] = index[0] + 1.0f;
      });
    });

    // Submit command group to queue to multiply matrices: c = a * b
    q.submit([&](auto &h) {
      // Read from a and b, write to c
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
      accessor c(c_buf, h, write_only);

      int width_a = a_buf.get_range()[1];

      // Execute kernel.
      h.parallel_for(range(M, P), [=](auto index) {
        // Get global position in Y direction.
        int row = index[0];
        // Get global position in X direction.
        int col = index[1];

        float sum = 0.0f;

        // Compute the result of one element of c
        for (int i = 0; i < width_a; i++) {
          sum += a[row][i] * b[i][col];
        }

        c[index] = sum;
      });
    });
  } catch (sycl::exception const &e) {
    cout << "An exception is caught while multiplying matrices.\n";
    terminate();
  }

  int result;
  cout << "Result of matrix multiplication using DPC++: ";
  result = VerifyResult(c_back);
  delete[] c_back;

  return result;
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
