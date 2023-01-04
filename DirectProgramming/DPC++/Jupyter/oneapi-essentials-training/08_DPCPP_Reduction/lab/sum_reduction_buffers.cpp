//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

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
      //# create accessors for buffer
      accessor acc_data(buf_data, h, read_only);

      //# nd-range kernel parallel_for with reduction parameter
      h.parallel_for(nd_range<1>{N, B}, reduction(buf_sum, h, plus<>()), [=](nd_item<1> it, auto& temp) {
        auto i = it.get_global_id(0);
        temp.combine(acc_data[i]);
      });
    });
  }
  std::cout << "Sum = " << sum << "\n";

  return 0;
}
