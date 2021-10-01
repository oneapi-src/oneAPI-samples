//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  std::vector<int> data(N);
  for (int i = 0; i < N; i++) data[i] = i;
  int sum = 0;
  {
    //# create buffers for data and sum
    buffer buf_data(data);
    buffer buf_sum(&sum, range(1));

    q.submit([&](handler& h) {
      //# create accessors for buffers
      accessor acc_data(buf_data, h, read_only);
      accessor acc_sum(buf_sum, h);

      //# nd-range kernel parallel_for with reduction parameter
      h.parallel_for(nd_range<1>{N, B}, ONEAPI::reduction(acc_sum, 0, ONEAPI::plus<>()), [=](nd_item<1> it, auto& temp) {
        auto i = it.get_global_id(0);
        temp.combine(acc_data[i]);
      });
    });
  }
  std::cout << "Sum = " << sum << "\n";

  return 0;
}

