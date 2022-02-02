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
  //# setup queue with in_order property
  queue q(property::queue::in_order{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# use parallel_for to calculate sum for work_group using reduce
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item){
    auto wg = item.get_group();
    auto i = item.get_global_id(0);

    //# Adds all elements in work_group using work_group reduce
    int sum_wg = reduce_over_group(wg, data[i], plus<>());

    //# write work_group sum to first location for each work_group
    if (item.get_local_id(0) == 0) data[i] = sum_wg;

  });

  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i+=B){
        sum += data[i];
    }
    data[0] = sum;
  }).wait();

  std::cout << "Sum = " << data[0] << "\n";

  free(data, q);
  return 0;
}

