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

  //# initialize inputs and outputs
  std::vector<int> data(N);
  for (int i = 0; i < N; i++) data[i] = i;
  int sum = 0, min = 0, max = 0;
  {
    //# create buffers
    buffer buf_data(data);
    buffer buf_sum(&sum, range(1));
    buffer buf_min(&min, range(1));
    buffer buf_max(&max, range(1));

    q.submit([&](handler& h) {
      //# create accessors for data and results
      accessor acc_data(buf_data, h, read_only);
        
      //# define reduction objects for sum, min, max reduction
      auto reduction_sum = reduction(buf_sum, h, plus<>());
      auto reduction_min = reduction(buf_min, h, minimum<>());
      auto reduction_max = reduction(buf_max, h, maximum<>());
      
      //# parallel_for with multiple reduction objects
      h.parallel_for(nd_range<1>{N, B}, reduction_sum, reduction_min, reduction_max, [=](nd_item<1> it, auto& temp_sum, auto& temp_min, auto& temp_max) {
        auto i = it.get_global_id();
        temp_sum.combine(acc_data[i]);
        temp_min.combine(acc_data[i]);
        temp_max.combine(acc_data[i]);
      });
    });
  }
 
  //# print results
  std::cout << "Sum       = " << sum << "\n";
  std::cout << "Min       = " << min << "\n"; 
  std::cout << "Max       = " << max << "\n";

  return 0;
}
