//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <string>

#ifndef NTIMES
#define NTIMES 10
#endif

class non_specialized_kernel;

constexpr int num_runs = NTIMES;
constexpr size_t scalar = 3;

cl_ulong triad(size_t array_size, size_t inner_loop_size) {
  cl_ulong min_time_ns0 = DBL_MAX;
  sycl::queue q = sycl::queue(sycl::property::queue::enable_profiling{});

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Setup arrays
  double *A0 = sycl::malloc_shared<double>(array_size / 2, q);
  double *B0 = sycl::malloc_shared<double>(array_size / 2, q);
  double *C0 = sycl::malloc_shared<double>(array_size / 2, q);

  for (size_t i = 0; i < array_size / 2; i++) {
    A0[i] = 1.0;
    B0[i] = 2.0;
    C0[i] = 0.0;
  }

  // Run main computation <num_runs> times & record best time
  for (size_t i = 0; i < num_runs; i++) {
    auto q0_event = q.submit([&](sycl::handler &h) {
      h.parallel_for<non_specialized_kernel>(array_size / 2, [=](auto idx) {
        // set trip count to runtime variable
        auto runtime_trip_count_const = inner_loop_size;
        auto accum = 0;
        for (size_t j = 0; j < runtime_trip_count_const; j++) {
          auto multiplier = scalar * j;
          accum = accum + A0[idx] + B0[idx] * multiplier;
        }
        C0[idx] = accum;
      });
    });

    q.wait();

    cl_ulong exec_time_ns0 =
        q0_event
            .get_profiling_info<sycl::info::event_profiling::command_end>() -
        q0_event
            .get_profiling_info<sycl::info::event_profiling::command_start>();

    std::cout << "Execution time (iteration " << i
              << ") [sec]: " << (double)exec_time_ns0 * 1.0E-9 << "\n";
    min_time_ns0 = std::min(min_time_ns0, exec_time_ns0);
  }

  // Check correctness
  bool error = false;
  for (size_t vi = 0; vi < array_size / 2; vi++) {
    // Compute test result
    auto vaccum = 0;
    auto vruntime_trip_count_const = inner_loop_size;
    for (size_t vj = 0; vj < vruntime_trip_count_const; vj++) {
      auto vmultiplier = scalar * vj;
      vaccum = vaccum + A0[vi] + B0[vi] * vmultiplier;
    }

    // Verify correctness of C0 for current index
    if (C0[vi] != vaccum) {
      std::cout << "\nResult incorrect (element " << vi << " is " << C0[vi]
                << ")!\n";
      error = true;
    }
  }

  // Release resources
  sycl::free(A0, q);
  sycl::free(B0, q);
  sycl::free(C0, q);

  if (error)
    return -1;

  std::cout << "Results are correct!\n\n";
  return min_time_ns0;
}

int main(int argc, char *argv[]) {

  // Input & program info display
  size_t array_size;
  size_t inner_loop_size;
  if (argc > 2) {
    array_size = std::stoi(argv[1]);
    inner_loop_size = std::stoi(argv[2]);
  } else {
    std::cout
        << "Run as ./<progname> <arraysize in elements> <inner loop size>\n";
    return 1;
  }
  std::cout << "Running with stream size of " << array_size << " elements ("
            << (array_size * sizeof(double)) / (double)1024 / 1024 << "MB)\n";

  std::cout << "Running with inner trip count of " << inner_loop_size << "\n";

  // Compute triad
  cl_ulong min_time = triad(array_size, inner_loop_size);
  size_t min_cmp = -1;

  if (min_time == min_cmp)
    return 1;

  size_t triad_bytes = 3 * sizeof(double) * array_size;
  std::cout << "Triad Bytes: " << triad_bytes << "\n";
  std::cout << "Time in sec (fastest run): " << min_time * 1.0E-9 << "\n";
  return 0;
}
