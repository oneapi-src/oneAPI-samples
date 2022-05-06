//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64;  // work-group size

int main() {
  queue q;
  std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";

  //# initialize data array using usm
  int *data = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) data[i] = i;
  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n\n";

  q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
    auto sg = item.get_sub_group();
    auto i = item.get_global_id(0);

    //# Add all elements in sub_group using sub_group collectives
    int result = reduce_over_group(sg, data[i], plus<>());

    //# write sub_group sum in first location for each sub_group
    if (sg.get_local_id()[0] == 0) {
      data[i] = result;
    } else {
      data[i] = 0;
    }
  }).wait();

  for (int i = 0; i < N; i++) std::cout << data[i] << " ";
  std::cout << "\n";

  free(data, q);
  return 0;
}
