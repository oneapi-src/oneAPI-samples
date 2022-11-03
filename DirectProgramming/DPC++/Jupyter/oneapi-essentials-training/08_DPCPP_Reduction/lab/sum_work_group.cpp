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
  //# setup queue with in_order property
  queue q(property::queue::in_order{});
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  auto data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;

  //# use parallel_for to calculate sum for each work_group
  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item){
    size_t index = item.get_global_id(0);
    if(item.get_local_id(0) == 0 ){
      int sum_wg = 0;
      for(int i=index; i<index+B; i++){
        sum_wg += data[i];
      }
      data[index] = sum_wg;
    }
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
