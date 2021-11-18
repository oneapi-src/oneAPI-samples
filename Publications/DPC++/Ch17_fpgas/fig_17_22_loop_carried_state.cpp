// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
using namespace sycl;

int generate_incremental_random_number(const int& state) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  constexpr int size = 64; 
  constexpr int seed = 0; 

  queue Q{ ext::intel::fpga_emulator_selector{} };

  buffer <int> B{ range{size} };

  Q.submit([&](handler &h){
      accessor output(B,h);

// BEGIN CODE SNIP
      h.single_task([=]() {
        int state = seed;
        for (int i=0; i < size; i++) {
          state = generate_incremental_random_number(state);
          output[i] = state;
        }
        });
// END CODE SNIP
      });

  return 0;
}

