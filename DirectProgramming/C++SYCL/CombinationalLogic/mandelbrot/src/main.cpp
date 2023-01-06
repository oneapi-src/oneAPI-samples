//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <chrono>
#include <iomanip>
#include <iostream>
#include <CL/sycl.hpp>
// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities/<version>/include/dpc_common.hpp
#include "dpc_common.hpp"
#include "mandel.hpp"


using namespace std;
using namespace sycl;

void ShowDevice(queue &q) {
  // Output platform and device information.
  auto device = q.get_device();
  auto p_name = device.get_platform().get_info<info::platform::name>();
  cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  auto p_version = device.get_platform().get_info<info::platform::version>();
  cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  auto d_name = device.get_info<info::device::name>();
  cout << std::setw(20) << "Device Name: " << d_name << "\n";
  auto max_work_group = device.get_info<info::device::max_work_group_size>();
  cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  auto max_compute_units = device.get_info<info::device::max_compute_units>();
  cout << std::setw(20) << "Max Compute Units: " << max_compute_units << "\n\n";
}

void Execute(queue &q) {
  // Demonstrate the Mandelbrot calculation serial and parallel.
#ifdef MANDELBROT_USM
  cout << "Parallel Mandelbrot set using USM.\n";
  MandelParallelUsm m_par(row_size, col_size, max_iterations, &q);
#else
  cout << "Parallel Mandelbrot set using buffers.\n";
  MandelParallel m_par(row_size, col_size, max_iterations);
#endif

  MandelSerial m_ser(row_size, col_size, max_iterations);

  // Run the code once to trigger JIT.
  m_par.Evaluate(q);

  // Run the parallel version and time it.
  dpc_common::TimeInterval t_par;
  for (int i = 0; i < repetitions; ++i) m_par.Evaluate(q);
  double parallel_time = t_par.Elapsed();

  // Print the results.
  m_par.Print();
  m_par.WriteImage();

  // Run the serial version.
  dpc_common::TimeInterval t_ser;
  m_ser.Evaluate();
  double serial_time = t_ser.Elapsed();

  // Report the results.
  cout << std::setw(20) << "Serial time: " << serial_time << "s\n";
  cout << std::setw(20) << "Parallel time: " << (parallel_time / repetitions)
       << "s\n";

  // Validate.
  m_par.Verify(m_ser);
}

int main(int argc, char *argv[]) {
  try {
    // Create a queue on the default device. Set SYCL_DEVICE_TYPE environment
    // variable to (CPU|GPU|FPGA|HOST) to change the device.
    queue q(default_selector_v);

    // Display the device info.
    ShowDevice(q);

    // Compute Mandelbrot set.
    Execute(q);
  } catch (...) {
    // Some other exception detected.
    cout << "Failed to compute Mandelbrot set.\n";
    std::terminate();
  }

  cout << "Successfully computed Mandelbrot set.\n";
  return 0;
}
