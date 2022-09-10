// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
#include <array>
using namespace sycl;

int main() {
  constexpr int count = 1024; 
  std::array<int, count> in_array;

  // Initialize input array
  for (int i=0; i < count; i++) { in_array[i] = i;}

  // Buffer initialized from in_array (std::array)
  buffer <int> B_in{ in_array };

  // Uninitialized buffer with count elements
  buffer <int> B_out{ range{count} };

  // Acquire queue to emulated FPGA device
  queue Q{ ext::intel::fpga_emulator_selector{} };

// BEGIN CODE SNIP
  // Create alias for pipe type so that consistent across uses
  using my_pipe = pipe<class some_pipe, int>;

  // ND-range kernel
  Q.submit([&](handler& h) {
      auto A = accessor(B_in, h);

      h.parallel_for(count, [=](auto idx) {
          my_pipe::write( A[idx] );
          });
      });

  // Single_task kernel
  Q.submit([&](handler& h) {
      auto A = accessor(B_out, h);

      h.single_task([=]() {
        for (int i=0; i < count; i++) {
          A[i] = my_pipe::read();
        }
      });
    });

// END CODE SNIP

  auto A = host_accessor(B_out);
  for (int i=0; i < count; i++) {
    if (A[i] != i) {
      std::cout << "Failure on element " << i << "\n";
      return 1;
    }
  }
  std::cout << "Passed!\n";
  return 0;
}

