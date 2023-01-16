//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// 1D HEAT TRANSFER: Using Data Parallel C++ Language to simulate 1D Heat
// Transfer.
//
// The code sample simulates the heat propagation according to the following
// equation (case where there is no heat generation):
//
//    dU/dt = k * d2U/dx2
//    (u(x,t+DT) - u(x,t)) / DT = k * (u(x+DX,t)- 2u(x,t) + u(x-DX,t)) / DX2
//    U(i) = C * (U(i+1) - 2 * U(i) + U(i-1)) + U(i)
//
// where constant C = k * dt / (dx * dx)
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// SYCL material used in this code sample:
//
// Basic structures of SYCL:
//   SYCL Queues (including device selectors and exception handlers)
//   SYCL Buffers and accessors (communicate data between the host and the
//   device)
//   SYCL Kernels (including parallel_for function and range<1> objects)
//
//******************************************************************************
// Content: (version 1.1)
//   1d_HeatTransfer.cpp
//
//******************************************************************************
#include <CL/sycl.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

constexpr float dt = 0.002f;
constexpr float dx = 0.01f;
constexpr float k = 0.025f;
constexpr float initial_temperature = 100.0f; // Initial temperature.

int failures = 0;

//
// Display input parameters used for this sample.
//
void Usage(const string &programName) {
  cout << " Incorrect parameters \n";
  cout << " Usage: ";
  cout << programName << " <n> <i>\n\n";
  cout << " n : Number of points to simulate \n";
  cout << " i : Number of timesteps \n";
}

//
// Initialize the temperature arrays
//
void Initialize(float *arr, float *arr_next, size_t num) {
  for (size_t i = 1; i < num; i++)
    arr[i] = arr_next[i] = 0.0f;
  arr[0] = arr_next[0] = initial_temperature;
}

//
// Compare host and device results
//
void CompareResults(string prefix, float *device_results, float *host_results,
                    size_t num_point, float C) {
  string path = prefix + "_error_diff.txt";
  float delta = 0.001f;
  float difference = 0.00f;
  double norm2 = 0;
  bool err = false;

  ofstream err_file;
  err_file.open(path);

  err_file << " \t idx\theat[i]\t\theat_CPU[i] \n";

  for (size_t i = 0; i < num_point + 2; i++) {
    err_file << " RESULT: " << i << "\t" << std::setw(12) << std::left
             << device_results[i] << "\t" << host_results[i] << "\n";

    difference = fabsf(host_results[i] - device_results[i]);
    norm2 += difference * difference;

    if (difference > delta) {
      err = true;
      err_file << ", diff: " << difference;
    }
  }

  if (err) {
    cout << "  FAIL! Please check " << path << "\n";
    failures++;
  } else
    cout << "  PASSED!\n";
}

//
// Compute heat on the device using SYCL buffer
//
void ComputeHeatBuffer(float C, size_t num_p, size_t num_iter, float *arr_CPU) {

  // Create a device queue using SYCL class queue
  queue q(default_selector_v);
  cout << "Using buffers\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>()
       << "\n";

  // Temperatures of the current and next iteration
  float *arr_host = new float[num_p + 2];
  float *arr_host_next = new float[num_p + 2];

  Initialize(arr_host, arr_host_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval t_par;

  auto *arr_buf = new buffer<float>(arr_host, range(num_p + 2));
  auto *arr_buf_next = new buffer<float>(arr_host_next, range(num_p + 2));

  // Iterate over timesteps
  for (size_t i = 0; i < num_iter; i++) {
    auto cg = [&](auto &h) {
      accessor arr(*arr_buf, h);
      accessor arr_next(*arr_buf_next, h);
      auto step = [=](id<1> idx) {
        size_t k = idx + 1;

        if (k == num_p + 1) {
          arr_next[k] = arr[k - 1];
        } else {
          arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
        }
      };

      h.parallel_for(range{num_p + 1}, step);
    };
    q.submit(cg);

    // Swap arrays for next step
    swap(arr_buf, arr_buf_next);
  }

  // Deleting will wait for tasks to complete and write data back to host
  // Write back is not needed for arr_buf_next
  arr_buf_next->set_write_back(false);
  delete arr_buf;
  delete arr_buf_next;

  // Display time used to process all time steps
  cout << "  Elapsed time: " << t_par.Elapsed() << " sec\n";

  CompareResults("buffer", ((num_iter % 2) == 0) ? arr_host : arr_host_next,
                 arr_CPU, num_p, C);

  delete[] arr_host;
  delete[] arr_host_next;
}

//
// Compute heat on the device using USM
//
void ComputeHeatUSM(float C, size_t num_p, size_t num_iter, float *arr_CPU) {
  // Timesteps depend on each other, so make the queue inorder
  property_list properties{property::queue::in_order()};

  // Create a device queue using DPC++ class queue
  queue q(default_selector_v, properties);
  cout << "Using USM\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>()
       << "\n";

  // Temperatures of the current and next iteration
  float *arr = malloc_shared<float>(num_p + 2, q);
  float *arr_next = malloc_shared<float>(num_p + 2, q);

  Initialize(arr, arr_next, num_p + 2);

  // Start timer
  dpc_common::TimeInterval time;

  // for each timesteps
  for (size_t i = 0; i < num_iter; i++) {
    auto step = [=](id<1> idx) {
      size_t k = idx + 1;
      if (k == num_p + 1)
        arr_next[k] = arr[k - 1];
      else
        arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];
    };

    q.parallel_for(range{num_p + 1}, step);

    // Swap arrays for next step
    swap(arr, arr_next);
  }

  // Wait for all the timesteps to complete
  q.wait_and_throw();

  // Display time used to process all time steps
  cout << "  Elapsed time: " << time.Elapsed() << " sec\n";

  CompareResults("usm", arr, arr_CPU, num_p, C);

  free(arr, q);
  free(arr_next, q);
}

//
// Compute heat serially on the host
//
float *ComputeHeatHostSerial(float *arr, float *arr_next, float C, size_t num_p,
                             size_t num_iter) {
  size_t i, k;

  // Set initial condition
  Initialize(arr, arr_next, num_p + 2);

  // Iterate over timesteps
  for (i = 0; i < num_iter; i++) {
    for (k = 1; k <= num_p; k++)
      arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];

    arr_next[num_p + 1] = arr[num_p];

    // Swap the buffers for the next step
    swap(arr, arr_next);
  }

  return arr;
}

int main(int argc, char *argv[]) {
  size_t n_point; // The number of points in 1D space
  size_t
      n_iteration; // The number of iterations to simulate the heat propagation

  // Read input parameters
  try {
    int np = stoi(argv[1]);
    int ni = stoi(argv[2]);
    if (np < 0 || ni < 0) {
      Usage(argv[0]);
      return -1;
    }
    n_point = np;
    n_iteration = ni;
  } catch (...) {
    Usage(argv[0]);
    return (-1);
  }

  cout << "Number of points: " << n_point << "\n";
  cout << "Number of iterations: " << n_iteration << "\n";

  // Temperatures of the current and next iteration
  float *heat_CPU = new float[n_point + 2];
  float *heat_CPU_next = new float[n_point + 2];

  // Constant used in the simulation
  float C = (k * dt) / (dx * dx);

  // Compute heat serially on CPU for comparision
  float *final_CPU = final_CPU =
      ComputeHeatHostSerial(heat_CPU, heat_CPU_next, C, n_point, n_iteration);

  try {
    ComputeHeatBuffer(C, n_point, n_iteration, final_CPU);
    ComputeHeatUSM(C, n_point, n_iteration, final_CPU);
  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
    failures++;
  }

  delete[] heat_CPU;
  delete[] heat_CPU_next;

  return failures;
}
