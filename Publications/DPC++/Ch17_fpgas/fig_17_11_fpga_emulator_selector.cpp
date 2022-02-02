// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp> // For fpga_selector
using namespace sycl;

void say_device (const queue& Q) {
  std::cout << "Device : " 
    << Q.get_device().get_info<info::device::name>() << "\n";
}

int main() {
  queue Q{ ext::intel::fpga_emulator_selector{} };
  say_device(Q);

  Q.submit([&](handler &h){
      h.parallel_for(1024, [=](auto idx) {
          // ...
          });
      });

  return 0;
}

