// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp> // For fpga_selector
using namespace sycl;

int generate_random_number_from_ID(const id<3>& I) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  queue Q{ INTEL::fpga_emulator_selector{} };

  buffer <int,3> B{ range{16,16,16} };

  Q.submit([&](handler &h){
      accessor output(B,h);
// BEGIN CODE SNIP
      h.parallel_for({16,16,16}, [=](auto I) {
        output[I] = generate_random_number_from_ID(I);
          });
// END CODE SNIP
      });

  return 0;
}

