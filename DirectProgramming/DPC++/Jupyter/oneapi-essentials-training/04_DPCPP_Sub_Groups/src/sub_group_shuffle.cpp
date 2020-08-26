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
  for (int i = 0; i < N; i++) data[i] = i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << std::endl << std::endl;

  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    intel::sub_group sg = item.get_sub_group();
    size_t i = item.get_global_id(0);

    //# swap adjasent items in array using sub_group shuffle_xor
    data[i] = sg.shuffle_xor(data[i], 1);
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  free(data, q);
  return 0;
}