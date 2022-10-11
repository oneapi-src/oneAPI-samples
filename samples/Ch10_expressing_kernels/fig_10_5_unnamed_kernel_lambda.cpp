// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
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

    queue Q{ default_selector_v };
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      // In many cases the explicit kernel name template parameter is not
      // required.
// BEGIN CODE SNIP
      h.parallel_for(size, [=](id<1> i) {
        data_acc[i] = data_acc[i] + 1;
      });
// END CODE SNIP
    });
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
