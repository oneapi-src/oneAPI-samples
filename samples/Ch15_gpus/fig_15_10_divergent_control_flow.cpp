// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr size_t array_size = 16;
  std::array<int, array_size> data;

  for (int i = 0; i < array_size; i++) {
    data[i] = i;
  }

  buffer dataBuf{data};

  queue Q{ default_selector_v };
  Q.submit([&](handler& h) {
      accessor dataAcc{ dataBuf, h };

      h.parallel_for(array_size, [=](id<1> i) {
          auto condition = i[0] & 1;
          if (condition) {
          dataAcc[i] = dataAcc[i] * 2; // odd
          } else {
          dataAcc[i] = dataAcc[i] + 1; // even
          }
          });
      });

  host_accessor dataAcc{ dataBuf };

  for (int i = 0; i < array_size; i++) {
    if (i & 1) {
      if (dataAcc[i] != i * 2) {
        std::cout << "Odd result did not validate at index " << i << "!\n";
        return -1;
      }
    } else {
      if (dataAcc[i] != i + 1) {
        std::cout << "Even result did not validate at index " << i << "!\n";
        return -1;
      }
    }
  }

  std::cout << "Success!\n";
  return 0;
}
