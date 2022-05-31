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

double triad(
    const std::vector<float>& vecA,
    const std::vector<float>& vecB,
    std::vector<float>& vecC ) {

  assert(vecA.size() == vecB.size() && vecB.size() == vecC.size());
  const size_t array_size = vecA.size();
  double min_time_ns = DBL_MAX;

  queue Q{ property::queue::enable_profiling{} };
  std::cout << "Running on device: " << Q.get_device().get_info<info::device::name>() << "\n";

  buffer<float> bufA(vecA);
  buffer<float> bufB(vecB);
  buffer<float> bufC(vecC);

  for (int i = 0; i< num_runs; i++) {
    auto Q_event = Q.submit([&](handler& h) {
        accessor A{ bufA, h };
        accessor B{ bufB, h };
        accessor C{ bufC, h };

        h.parallel_for(array_size, [=](id<1> idx) {
            C[idx] = A[idx] + B[idx] * scalar;
            });
        });

    double exec_time_ns =
      Q_event.get_profiling_info<info::event_profiling::command_end>() -
      Q_event.get_profiling_info<info::event_profiling::command_start>();

    std::cout << "Execution time (iteration " << i << ") [sec]: " << (double)exec_time_ns * 1.0E-9 << "\n";
    min_time_ns = std::min(min_time_ns, exec_time_ns);
  }

  return min_time_ns;
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
    << " elements (" << (array_size * sizeof(float))/(double)1024/1024 << "MB)\n";

  std::vector<float> D(array_size, 1.0);
  std::vector<float> E(array_size, 2.0);
  std::vector<float> F(array_size, 0.0);

  double min_time = triad(D, E, F);

  // Check correctness
  for ( int i = 0; i < array_size; i++) {
    if (F[i] != D[i] + scalar * E[i]) {
      std::cout << "\nResult incorrect (element " << i << " is " << F[i] << ")!\n";
      return 1;
    }
  }
  std::cout << "Results are correct!\n\n";

  size_t triad_bytes = 3 * sizeof(float) * array_size;
  std::cout << "Triad Bytes: " << triad_bytes << "\n";
  std::cout << "Time in sec (fastest run): " << min_time * 1.0E-9 << "\n";

  double triad_bandwidth = 1.0E-09 * triad_bytes/(min_time*1.0E-9);
  std::cout << "Bandwidth of fastest run in GB/s: " << triad_bandwidth << "\n";

  return 0;
}

