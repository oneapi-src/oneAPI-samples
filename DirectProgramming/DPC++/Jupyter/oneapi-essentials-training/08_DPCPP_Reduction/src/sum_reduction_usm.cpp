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

  //# implicit USM for writing sum value
  int* sum = malloc_shared<int>(1, q);
  *sum = 0;

  //# nd-range kernel parallel_for with reduction parameter
  q.parallel_for(nd_range<1>{N, B}, reduction(sum, plus<>()), [=](nd_item<1> it, auto& temp) {
    auto i = it.get_global_id(0);
    temp.combine(data[i]);
  }).wait();

  std::cout << "Sum = " << *sum << "\n";

  free(data, q);
  free(sum, q);
  return 0;
}

