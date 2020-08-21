//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <iomanip>  // setprecision library
#include <iostream>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"



// cpu_seq_calc_pi is a simple sequential CPU routine
// that calculates all the slices and then
// does a reduction.
float cpu_seq_calc_pi(int num_steps) {
  float step = 1.0 / (float)num_steps;
  float x;
  float pi;
  float sum = 0.0;
  for (int i = 1; i < num_steps; i++) {
    x = ((float)i - 0.5f) * step;
    sum = sum + 4.0f / (1.0f + x * x);
  }
  pi = sum * step;
  return pi;
}

// openmp_host_calc_pi is a simple parallel
// calcuation that uses openmp running
// on the host.   By default openmp
// will use all the cores available
// and execute the code in parallel and
// then perform a reduction.
float openmp_host_calc_pi(int num_steps) {
  float step = (1.0f / num_steps);
  float pi = 0.0;
  float sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 1; i < num_steps; i++) {
    float x = ((float)i - 0.5f) * step;
    sum = sum + 4.0f / (1.0f + x * x);
  }
  pi = step * sum;
  return pi;
}

// openmp_device_calc_pi is a simple parallel
// calcuation that uses openmp running
// on the device through the use of the
// target specifier.
// This will  execute the code in parallel.

float openmp_device_calc_pi(int num_steps) {
  float pi = 0.0;
  float step = (1.0f / num_steps);
  float sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+ : sum)
  for (int i = 1; i < num_steps; i++) {
    float x = ((float)i - 0.5f) * step;
    sum = sum + 4.0f / (1.0 + x * x);
  }
  pi = sum * step;
  return pi;
}

int main(int argc, char** argv) {
  int num_steps = 1000000;
  printf("Number of steps is %d\n", num_steps);
  float pi;

  // Due to the overhead associated with
  // JIT, run the offload calculation once
  // that allows code to be compiled.  Execution
  // time is measured the 2nd time you run it.
  pi = openmp_device_calc_pi(num_steps);

  dpc_common::TimeInterval T;
  pi = cpu_seq_calc_pi(num_steps);
  auto stop = T.Elapsed();
  std::cout << "Cpu Seq calc: \t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop << " seconds"
            << "\n";

  dpc_common::TimeInterval T2;
  pi = openmp_host_calc_pi(num_steps);
  auto stop2 = T2.Elapsed();
  std::cout << "Host OpenMP:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop2 << " seconds"
            << "\n";

  dpc_common::TimeInterval T3;
  pi = openmp_device_calc_pi(num_steps);
  auto stop3 = T3.Elapsed();
  std::cout << "Offload OpenMP:\t\t";
  std::cout << std::setprecision(3) << "PI =" << pi;
  std::cout << " in " << stop3 << " seconds"
            << "\n";

  std::cout << "success\n";
  return 0;
}
