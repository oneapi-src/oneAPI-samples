// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

int main() {
  constexpr int size = 16;
  std::array<int, size> data;
  buffer B{ data };

  queue Q{};  // Select any device for this queue

  std::cout << "Selected device is: " <<
    Q.get_device().get_info<info::device::name>() << "\n";

  Q.submit([&](handler& h) {
    accessor acc{B, h};
    h.parallel_for(size , [=](auto& idx) {
      acc[idx] = idx;
    });
  });

  return 0;
}

