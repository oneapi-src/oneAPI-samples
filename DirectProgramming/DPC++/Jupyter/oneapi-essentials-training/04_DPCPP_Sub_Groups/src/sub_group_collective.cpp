//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static const size_t N = 256; // global size
static const size_t B = 64;  // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;

  //# initialize data array using usm
  int *data = static_cast<int *>(malloc_shared(N * sizeof(int), q));
  for (int i = 0; i < N; i++) data[i] = 1 + i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << std::endl << std::endl;

  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    intel::sub_group sg = item.get_sub_group();
    size_t i = item.get_global_id(0);

    //# Adds all elements in sub_group using sub_group collectives
    int sum = reduce(sg, data[i], intel::plus<>());

    //# write sub_group sum in first location for each sub_group
    if (sg.get_local_id()[0] == 0) {
      data[i] = sum;
    } else {
      data[i] = 0;
    }
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  free(data, q);
  return 0;
}
