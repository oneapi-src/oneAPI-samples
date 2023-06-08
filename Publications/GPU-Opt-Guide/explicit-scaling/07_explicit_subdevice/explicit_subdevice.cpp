//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// clang-format off
// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <string>
using namespace sycl;

constexpr int num_runs = 10;
constexpr size_t scalar = 3;

cl_ulong triad(size_t array_size) {

  cl_ulong min_time_ns0 = DBL_MAX;
  cl_ulong min_time_ns1 = DBL_MAX;

  device dev = device(gpu_selector_v);

  std::vector<device> subdev = {};
  subdev = dev.create_sub_devices<sycl::info::partition_property::
    partition_by_affinity_domain>(sycl::info::partition_affinity_domain::numa);

  queue q[2] = {queue(subdev[0], property::queue::enable_profiling{}),
    queue(subdev[1], property::queue::enable_profiling{})};

  std::cout << "Running on device: " <<
    q[0].get_device().get_info<info::device::name>() << "\n";
  std::cout << "Running on device: " <<
    q[1].get_device().get_info<info::device::name>() << "\n";

  double *A0 = malloc_shared<double>(array_size/2 * sizeof(double), q[0]);
  double *B0 = malloc_shared<double>(array_size/2 * sizeof(double), q[0]);
  double *C0 = malloc_shared<double>(array_size/2 * sizeof(double), q[0]);

  double *A1 = malloc_shared<double>(array_size/2 * sizeof(double), q[1]);
  double *B1 = malloc_shared<double>(array_size/2 * sizeof(double), q[1]);
  double *C1 = malloc_shared<double>(array_size/2 * sizeof(double), q[1]);

  for ( int i = 0; i < array_size/2; i++) {
     A0[i]= 1.0; B0[i]= 2.0; C0[i]= 0.0;
     A1[i]= 1.0; B1[i]= 2.0; C1[i]= 0.0;
  }

  for (int i = 0; i< num_runs; i++) {
    auto q0_event = q[0].submit([&](handler& h) {
        h.parallel_for(array_size/2, [=](id<1> idx) {
            C0[idx] = A0[idx] + B0[idx] * scalar;
            });
        });

    auto q1_event = q[1].submit([&](handler& h) {
        h.parallel_for(array_size/2, [=](id<1> idx) {
            C1[idx] = A1[idx] + B1[idx] * scalar;
            });
        });

    q[0].wait();
    q[1].wait();

    cl_ulong exec_time_ns0 =
      q0_event.get_profiling_info<info::event_profiling::command_end>() -
      q0_event.get_profiling_info<info::event_profiling::command_start>();

    std::cout << "Tile-0 Execution time (iteration " << i << ") [sec]: "
      << (double)exec_time_ns0 * 1.0E-9 << "\n";
    min_time_ns0 = std::min(min_time_ns0, exec_time_ns0);

    cl_ulong exec_time_ns1 =
      q1_event.get_profiling_info<info::event_profiling::command_end>() -
      q1_event.get_profiling_info<info::event_profiling::command_start>();

    std::cout << "Tile-1 Execution time (iteration " << i << ") [sec]: "
      << (double)exec_time_ns1 * 1.0E-9 << "\n";
    min_time_ns1 = std::min(min_time_ns1, exec_time_ns1);
  }

  // Check correctness
  bool error = false;
  for ( int i = 0; i < array_size/2; i++) {
    if ((C0[i] != A0[i] + scalar * B0[i]) || (C1[i] != A1[i] + scalar * B1[i])) {
      std::cout << "\nResult incorrect (element " << i << " is " << C0[i] << ")!\n";
      error = true;
    }
  }

  sycl::free(A0, q[0]);
  sycl::free(B0, q[0]);
  sycl::free(C0, q[0]);

  sycl::free(A1, q[1]);
  sycl::free(B1, q[1]);
  sycl::free(C1, q[1]);

  if (error) return -1;

  std::cout << "Results are correct!\n\n";
  return std::max(min_time_ns0, min_time_ns1);
}

int main(int argc, char *argv[]) {

  size_t array_size;
  if (argc > 1 ) {
    array_size =  std::stoi(argv[1]);
  }
  else {
    std::cout << "Run as ./<progname> <arraysize in elements>\n";
    return 1;
  }
  std::cout << "Running with stream size of " << array_size
    << " elements (" << (array_size * sizeof(double))/(double)1024/1024 << "MB)\n";

  cl_ulong min_time = triad(array_size);

  if (min_time == -1) return 1;
  size_t triad_bytes = 3 * sizeof(double) * array_size;
  std::cout << "Triad Bytes: " << triad_bytes << "\n";
  std::cout << "Time in sec (fastest run): " << min_time * 1.0E-9 << "\n";
  double triad_bandwidth = 1.0E-09 * triad_bytes/(min_time*1.0E-9);
  std::cout << "Bandwidth of fastest run in GB/s: " << triad_bandwidth << "\n";
  return 0;
}
