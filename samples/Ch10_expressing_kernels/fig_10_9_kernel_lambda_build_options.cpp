// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
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

    queue Q{ cpu_selector_v };
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

// BEGIN CODE SNIP
    kernel_id Add_KID = get_kernel_id<class Add>();
    auto kb = get_kernel_bundle<bundle_state::executable>(Q.get_context(), {Add_KID});
    kernel k = kb.get_kernel(Add_KID);

    Q.submit([&](handler& h) {
      accessor data_acc {data_buf, h};

      h.parallel_for<class Add>(
          // This uses the previously compiled kernel k, the body of which is
          // defined here as a lambda
          k,
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
