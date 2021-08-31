// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

class Add;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

    queue Q{ cpu_selector{} };
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

// BEGIN CODE SNIP
    // This compiles the kernel named by the specified template
    // parameter using the "fast relaxed math" build option.
    program p(Q.get_context());

    p.build_with_kernel_type<class Add>("-cl-fast-relaxed-math");

    Q.submit([&](handler& h) {
      accessor data_acc {data_buf, h};

      h.parallel_for<class Add>(
          // This uses the previously compiled kernel.
          p.get_kernel<class Add>(),
          range{size},
          [=](id<1> i) {
            data_acc[i] = data_acc[i] + 1;
          });
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
