// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++)
    data[i] = i;

  {
    buffer dataBuf{data};

    queue Q{ default_selector_v };
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

    Q.submit([&](handler& h) {
// BEGIN CODE SNIP
      // This is a typical global accessor.
      accessor dataAcc {dataBuf, h};

      // This is a 1D local accessor consisting of 16 ints:
      auto localIntAcc = local_accessor<int, 1>(16, h);

      // This is a 2D local accessor consisting of 4 x 4 floats:
      auto localFloatAcc = local_accessor<float, 2>({4, 4}, h);

      h.parallel_for(nd_range<1>{{size}, {16}}, [=](nd_item<1> item) {
        auto index = item.get_global_id();
        auto local_index = item.get_local_id();

        // Within a kernel, a local accessor may be read from
        // and written to like any other accessor.
        localIntAcc[local_index] = dataAcc[index] + 1;
        dataAcc[index] = localIntAcc[local_index];
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
