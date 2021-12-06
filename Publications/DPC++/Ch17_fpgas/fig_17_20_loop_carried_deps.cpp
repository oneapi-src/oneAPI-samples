// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
using namespace sycl;

int generate_random_number(const int& state) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  constexpr int size = 64; 
  queue Q{ ext::intel::fpga_emulator_selector{} };

  buffer <int> B{ range{size} };

  Q.submit([&](handler &h){
      accessor output(B,h);

      h.single_task([=]() {
// BEGIN CODE SNIP
        int a = 0;
        for (int i=0; i < size; i++) {
          a = a + i;
        }
// END CODE SNIP
          });
      });

  return 0;
}

