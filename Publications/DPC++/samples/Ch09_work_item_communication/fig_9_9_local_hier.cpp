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
    buffer data_buf{data};

    queue Q;
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

    Q.submit([&](handler& h) {
      // This is a typical global accessor.
      accessor data_acc {data_buf, h};

// BEGIN CODE SNIP
      range group_size{16};
      range num_groups = size / group_size;

      h.parallel_for_work_group(num_groups, group_size, [=](group<1> group) {
        // This variable is declared at work-group scope, so
        // it is allocated in local memory and accessible to
        // all work-items.
        int localIntArr[16];

        // There is an implicit barrier between code and variables
        // declared at work-group scope and the code and variables
        // at work-item scope.

        group.parallel_for_work_item([&](h_item<1> item) {
          auto index = item.get_global_id();
          auto local_index = item.get_local_id();

          // The code at work-item scope can read and write the
          // variables declared at work-group scope.
          localIntArr[local_index] = index + 1;
          data_acc[index] = localIntArr[local_index];
        });
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
