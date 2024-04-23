//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 1024; // global size
static constexpr size_t B = 128; // work-group size
static constexpr size_t S = 32; // sub_group size

int main() {
  //# setup queue
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# use parallel_for and sub_groups to calculate sum
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item)[[intel::reqd_sub_group_size(S)]] {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# Adds all elements in sub_group using sub_group reduce
    int sum_sg = reduce_over_group(sg, data[i], plus<>());

    //# write sub_group sum to first location for each sub_group
    if (sg.get_local_id()[0] == 0) data[i] = sum_sg;

  });

  q.single_task([=](){
    int sum = 0;
    for(int i=0;i<N;i+=S){
        sum += data[i];
    }
    data[0] = sum;
  });

  std::cout << "Sum = " << data[0] << "\n";
  
  free(data, q);
  return 0;
}


