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
  //# setup queue with default selector
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# implicit USM for writing min and max value
  int* min = malloc_shared<int>(1, q);
  int* max = malloc_shared<int>(1, q);
  *min = 0;
  *max = 0;
    
  //# STEP 1 : Create reduction objects for computing min and max
  auto reduction_min = reduction(min, minimum<>());
  auto reduction_max = reduction(max, maximum<>());

  //# Reduction Kernel get min and max
  q.submit([&](handler& h) {
      
    //# STEP 2 : add parallel_for with reduction objects for min and max
    h.parallel_for(nd_range<1>{N, B}, reduction_min, reduction_max, [=](nd_item<1> it, auto& temp_min, auto& temp_max) {
      auto i = it.get_global_id(0);
      temp_min.combine(data[i]);
      temp_max.combine(data[i]);
    });
  }).wait();
    
  //# STEP 3 : Compute mid_range from min and max
  int mid_range = 0
  mid_range = (min[0] + max[0])/2;
  std::cout << "Mid-Range = " << mid_range << "\n";

  free(data, q);
  free(min, q);
  free(max, q);
  return 0;
}
