//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// 1D HEAT TRANSFER: Using Intel® oneAPI DPC++ Language to simulate 1D Heat
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
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide
// and search based on relevant terms noted in the comments.
//
// DPC++ material used in this code sample:
//
// Basic structures of DPC++:
//   DPC++ Queues (including device selectors and exception handlers)
//   DPC++ Buffers and accessors (communicate data between the host and the
//   device)
//   DPC++ Kernels (including parallel_for function and range<1> objects)
//
//******************************************************************************
// Content: (version 1.1)
//   1d_HeatTransfer.cpp
//
//******************************************************************************
#include <CL/sycl.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace sycl;
using namespace std;

constexpr float dt = 0.002f;
constexpr float dx = 0.01f;
constexpr float k = 0.025f;
constexpr float initial_temperature = 100.0f;  // Initial temperature.

//
// Display input parameters used for this sample.
//
void Usage(string programName) {
  cout << " Incorrect parameters \n";
  cout << " Usage: ";
  cout << programName << " <n> <i>\n\n";
  cout << " n : Number of points to simulate \n";
  cout << " i : Number of timesteps \n";
}

//
// Initialize the temperature arrays
//
void Initialize(float* arr, float* arr_next, unsigned int num) {
  arr[0] = arr_next[0] = initial_temperature;
  for (unsigned int i = 0; i < num; i++)
    arr[i] = arr_next[i] = 0.0f;
}

//
// Compare host and device results
//
bool CompareResults(string prefix, float* device_results, float* host_results,
                    unsigned int num_point, float C) {
  string path = prefix + "_error_diff.txt";
  float delta = 0.001f;
  float difference = 0.00f;
  double norm2 = 0;
  bool err = false;

  ofstream err_file;
  err_file.open(path);

  err_file << " \t idx\theat[i]\t\theat_CPU[i] \n";

  for (unsigned int i = 0; i < num_point + 2; i++) {
    err_file << "\n RESULT: " << i << "\t" << std::setw(12) << std::left
             << device_results[i] << "\t" << host_results[i];

    difference = fabsf(host_results[i] - device_results[i]);
    norm2 += difference * difference;

    if (difference > delta) {
      err = true;
      err_file << ", diff: " << difference;
    }
  }

  if (err == true)
    cout << "  FAIL! Please check " << path << "\n";
  else
    cout << "  PASSED!\n";
  return err;
}

//
// Compute heat on the device
//
void ComputeHeatDevice(float C, unsigned int num_p, unsigned int num_iter,
		       float* arr_CPU) {
  // Define the device queue
  queue q;
  cout << "Device\n";
  cout << "  Kernel runs on " << q.get_device().get_info<info::device::name>() << "\n";

  // Temperatures of the current and next iteration
  float* arr = new float[num_p + 2];
  float* arr_next = new float[num_p + 2];

  Initialize(arr, arr_next, num_p + 2);

  buffer temperature_buf(arr, range(num_p + 2));
  buffer temperature_next_buf(arr_next, range(num_p + 2));

  // Start timer
  dpc_common::TimeInterval t_par;

  // Iterate over timesteps
  for (int i = 1; i <= num_iter; i++) {
    q.submit([&](auto& h) {
	       // The size of memory amount that will be given to the buffer.
	       range num_items{num_p + 2};
	       
	       accessor temperature(i%2 ? temperature_buf : temperature_next_buf, h);
	       accessor temperature_next(i%2 ? temperature_next_buf : temperature_buf, h);

	       h.parallel_for(num_items,
			      [=](id<1> k) {

				if (k == 0) {
				} else if (k == num_p + 1) {
				  temperature_next[k] = temperature[k - 1];
				} else {
				  temperature_next[k] =
				    C * (temperature[k + 1] - 2 * temperature[k] + temperature[k - 1]) +
				    temperature[k];
				}
			      });  // end parallel for loop in kernel1
	     });    // end device queue
  }          // end iteration

  // Wait for all tasks to complete
  q.wait_and_throw();

  // Display time used to process all time steps
  cout << "  Elapsed time: " << t_par.Elapsed() << " sec\n";

  CompareResults("parallel", num_iter % 2 == 0 ? arr : arr_next, arr_CPU, num_p, C);

  delete [] arr;
  delete [] arr_next;
}


//
// Compute heat serially on the host
//
float* ComputeHeatHostSerial(float* arr, float* arr_next, float C,
                             unsigned int num_p, unsigned int num_iter) {
  unsigned int i, k;
  float* swap;

  // Set initial condition
  Initialize(arr, arr_next, num_p + 2);

  // Iterate over timesteps
  for (i = 1; i <= num_iter; i++) {
    for (k = 1; k <= num_p; k++)
      arr_next[k] = C * (arr[k + 1] - 2 * arr[k] + arr[k - 1]) + arr[k];

    arr_next[num_p + 1] = arr[num_p];

    // Swap the buffers at every iteration.
    swap = arr;
    arr = arr_next;
    arr_next = swap;
  }

  return arr;
}


int main(int argc, char* argv[]) {
  unsigned int n_point;      // The number of point in 1D space
  unsigned int n_iteration;  // The number of iteration to simulate the heat propagation

  // Read input parameters
  try {
    n_point = stoi(argv[1]);
    n_iteration = stoi(argv[2]);

  } catch (...) {
    Usage(argv[0]);
    return (-1);
  }

  cout << "Number of points: " << n_point << "\n";
  cout << "Number of iterations: " << n_iteration << "\n";

  // Constant used in the simulation
  float C = (k * dt) / (dx * dx);

  // Temperatures of the current and next iteration
  float* heat_CPU = new float[n_point + 2];
  float* heat_CPU_next = new float[n_point + 2];

  // Compute heat serially on CPU for comparision
  float* final_CPU = 
    final_CPU = ComputeHeatHostSerial(heat_CPU, heat_CPU_next, C, n_point, n_iteration);

  try {
    ComputeHeatDevice(C, n_point, n_iteration, final_CPU);
  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
  }

  delete[] heat_CPU;
  delete[] heat_CPU_next;

  return 0;
}
