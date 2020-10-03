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
constexpr float temp = 100.0f;  // Initial temperature.

//************************************
// Function description: display input parameters used for this sample.
//************************************
void Usage(string programName) {
  cout << " Incorrect parameters \n";
  cout << " Usage: ";
  cout << programName << " <n> <i>\n\n";
  cout << " n : Number of points to simulate \n";
  cout << " i : Number of timesteps \n";
}

//************************************
// Function description: initialize the array.
//************************************
void Initialize(float* array, unsigned int num, unsigned int idx) {
  for (unsigned int i = idx; i < num; i++) array[i] = 0.0f;
}

//************************************
// Function description: compute the heat in the device (in parallel).
//************************************
float* ComputeHeatDeviceParallel(float* arr, float* arr_next, float C,
                                 unsigned int num_p, unsigned int num_iter,
                                 float temp) {
  unsigned int i;

  try {
    // Define the device queue
    queue q = default_selector{};
    cout << "Kernel runs on " << q.get_device().get_info<info::device::name>()
         << "\n";

    // Set boundary condition at one end.
    arr[0] = arr_next[0] = temp;

    float* current_data_ptr = arr;
    float* next_data_ptr = arr_next;

    // Buffer scope
    {
      buffer temperature_buf(current_data_ptr, range(num_p + 2));
      buffer temperature_next_buf(next_data_ptr, range(num_p + 2));

      // Iterate over timesteps
      for (i = 1; i <= num_iter; i++) {
        if (i % 2 != 0) {
          q.submit([&](auto& h) {
            // The size of memory amount that will be given to the buffer.
            range<1> num_items{num_p + 2};

            accessor temperature(temperature_buf, h);
            accessor temperature_next(temperature_next_buf, h);

            h.parallel_for(num_items, [=](id<1> k) {
              size_t gid = k.get(0);

              if (gid == 0) {
              } else if (gid == num_p + 1) {
                temperature_next[k] = temperature[k - 1];
              } else {
                temperature_next[k] =
                    C * (temperature[k + 1] - 2 * temperature[k] + temperature[k - 1]) +
                    temperature[k];
              }
            });  // end parallel for loop in kernel1
          });    // end device queue

        } else {
          q.submit([&](handler& h) {
            // The size of memory amount that will be given to the buffer.
            range<1> num_items{num_p + 2};

            accessor temperature(temperature_buf, h);
            accessor temperature_next(temperature_next_buf, h);

            h.parallel_for(num_items, [=](id<1> k) {
              size_t gid = k.get(0);

              if (gid == 0) {
              } else if (gid == num_p + 1) {
                temperature[k] = temperature_next[k - 1];
              } else {
                temperature[k] = C * (temperature_next[k + 1] - 2 * temperature_next[k] +
                                  temperature_next[k - 1]) +
                             temperature_next[k];
              }
            });  // end parallel for loop in kernel2
          });    // end device queue
        }        // end if %2
      }          // end iteration
    }            // end buffer scope

    q.wait_and_throw();

  } catch (sycl::exception e) {
    cout << "SYCL exception caught: " << e.what() << "\n";
  }

  if (i % 2 != 0)
    return arr;
  else
    return arr_next;
}

//************************************
// Function description: compute the heat in the host (in serial).
//************************************
float* ComputeHeatHostSerial(float* arr, float* arr_next, float C,
                             unsigned int num_p, unsigned int num_iter,
                             float temp) {
  unsigned int i, k;
  float* swap;

  // Set initial condition
  Initialize(arr, num_p + 2, 0);
  Initialize(arr_next, num_p + 2, 0);

  // Set boundary condition at one end.
  arr[0] = arr_next[0] = temp;

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

//************************************
// Function description: calculate the results computed by the host and by the
// device.
//************************************
bool CompareResults(float* device_results, float* host_results,
                    unsigned int num_point, float C) {
  float delta = 0.001f;
  float difference = 0.00f;
  double norm2 = 0;
  bool err = false;

  ofstream err_file;
  err_file.open("error_diff.txt");

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

  return err;
}

int main(int argc, char* argv[]) {
  unsigned int n_point;  // The number of point in 1D space
  unsigned int
      n_iteration;  // The number of iteration to simulate the heat propagation

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

  // Array heat and heat_next arrays store temperatures of the current and next
  // iteration of n_point (calculated in kernel)
  float* heat = new float[n_point + 2];
  float* heat_next = new float[n_point + 2];

  // heat_CPU and heat_next_CPU store temperatures of the current and next
  // iteration of n_point (calculated in CPU or comparison)
  float* heat_CPU = new float[n_point + 2];
  float* heat_CPU_next = new float[n_point + 2];

  // Constant used in the simulation
  float C = (k * dt) / (dx * dx);

  // Heat initial condition at t = 0
  Initialize(heat, n_point + 2, 0);
  Initialize(heat_next, n_point + 2, 0);

  // Start timer
  dpc_common::TimeInterval t_par;

  float* final_device =
      ComputeHeatDeviceParallel(heat, heat_next, C, n_point, n_iteration, temp);

  // Display time used by device
  cout << "Elapsed time: " << t_par.Elapsed() << " sec\n";

  // Compute heat in CPU in (for comparision)
  float* final_CPU = NULL;

  final_CPU = ComputeHeatHostSerial(heat_CPU, heat_CPU_next, C, n_point,
                                    n_iteration, temp);

  // Compare the results computed in device (in parallel) and in host (in
  // serial)
  bool err = CompareResults(final_device, final_CPU, n_point, C);

  if (err == true)
    cout << "Please check the error_diff.txt file ...\n";
  else
    cout << "PASSED! There is no difference between the results computed "
            "in host and in kernel.\n";

  // Cleanup
  delete[] heat;
  delete[] heat_next;
  delete[] heat_CPU;
  delete[] heat_CPU_next;

  return 0;
}
