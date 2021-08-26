// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

// BEGIN CODE SNIP
    // Note: This must select a device that supports interop!
    queue Q{ cpu_selector{} };

    program p{Q.get_context()};
    p.build_with_source(R"CLC(
            kernel void add(global int* data) {
                int index = get_global_id(0);
                data[index] = data[index] + 1;
            }
        )CLC",
        "-cl-fast-relaxed-math");

    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

    Q.submit([&](handler& h) {
      accessor data_acc {data_buf, h};

      h.set_args(data_acc);
      h.parallel_for(size, p.get_kernel("add"));
    });
// END CODE SNIP
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
