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
  cout << "Initializing" << "\n";
  // Host memory buffer that device will write data back before destruction.
  float(*a_back)[N] = new float[M][N];
  float(*b_back)[P] = new float[N][P];
  float(*c_back)[P] = new float[M][P];

  // Intialize a_back
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) a_back[i][j] = 1.0f;

  // Intialize b_back
  for (int i = 0; i < N; i++)
    for (int j = 0; j < P; j++) b_back[i][j] = i + 1.0f;

  // Intialize c_back
  for (int i = 0; i < M; i++)
    for (int j = 0; j < P; j++) c_back[i][j] = 0.0f;

  // Initialize the device queue with the default selector. The device queue is
  // used to enqueue kernels. It encapsulates all states needed for execution.
  {
    property_list propList = property_list{property::queue::enable_profiling()};

    queue q(default_selector_v);

    cout << "Computing" << "\n";
    cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    cout << "Device compute units: " << q.get_device().get_info<info::device::max_compute_units>() << "\n";
    auto maxWorkItemSize = q.get_device().get_info<info::device::max_work_item_sizes<3>>();
    cout << "Device max work item size: " << maxWorkItemSize.get(0) << ", " << maxWorkItemSize.get(1) << ", " << maxWorkItemSize.get(2) << "\n";
    cout << "Device max work group size: " << q.get_device().get_info<info::device::max_work_group_size>() << "\n";

    // Create 2D buffers for matrices, buffer c is bound with host memory c_back
    float * dev_a = sycl::malloc_device<float>(M*N, q);
    float * dev_b = sycl::malloc_device<float>(N*P, q);
    float * dev_c = sycl::malloc_device<float>(M*P, q);

    cout << "Problem size: c(" << M << "," << P << ") = a(" << M << "," << N
         << ") * b(" << N << "," << P << ")\n";

    // Using three command groups to illustrate execution order. The use of
    // first two command groups for initializing matrices is not the most
    // efficient way. It just demonstrates the implicit multiple command group
    // execution ordering.

    // Submit command group to queue to initialize matrix a
    q.memcpy(dev_a, &a_back[0], M*N * sizeof(float));

    // Submit command group to queue to initialize matrix b
    q.memcpy(dev_b, &b_back[0], N*P * sizeof(float));

    // Submit command group to queue to initialize matrix c
    q.submit([&](auto &h) {
        h.memcpy(dev_c, &c_back[0], M*P * sizeof(float));
    });

    q.wait();

    // Submit command group to queue to multiply matrices: c = a * b
    q.submit([&](auto &h) {
      // Read from a and b, write to c
      int width_a = N;

      // Execute kernel.
      h.parallel_for(range(M, P), [=](auto index) {
        // Get global position in Y direction.
        int row = index[0];  // m
        int col = index[1];  // p
        float sum = 0.0f;

        // Compute the result of one element of c
        for (int i = 0; i < width_a; i++) {
	  auto a_index = row * width_a + i;
	  auto b_index = i * P + col;
          sum += dev_a[a_index] * dev_b[b_index];
        }
      
	auto idx = row * P + col;
        dev_c[idx] = sum;
      });
    });
  
    q.wait();

    q.memcpy(&c_back[0], dev_c, M*P * sizeof(float));

    q.wait();
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
