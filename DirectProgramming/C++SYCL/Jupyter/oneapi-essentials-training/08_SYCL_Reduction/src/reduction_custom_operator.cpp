//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <time.h>

using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64; // work-group size

template <typename T, typename I>
struct pair {
  bool operator<(const pair& o) const {
    return val <= o.val || (val == o.val && idx <= o.idx);
  }
  T val;
  I idx;
};

int main() {
  //# setup queue with default selector
  queue q;
 
  //# initialize input data and result using usm
  auto result = malloc_shared<pair<int, int>>(1, q);
  auto data = malloc_shared<int>(N, q);

  //# initialize input data with random numbers
  srand(time(0));
  for (int i = 0; i < N; ++i) data[i] = rand() % 256;
  std::cout << "Input Data:\n";
  for (int i = 0; i < N; i++) std::cout << data[i] << " "; std::cout << "\n\n";

  //# custom operator for reduction to find minumum and index
  pair<int, int> operator_identity = {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()};
  *result = operator_identity;
  auto reduction_object = reduction(result, operator_identity, minimum<pair<int, int>>());

  //# parallel_for with user defined reduction object
  q.parallel_for(nd_range<1>{N, B}, reduction_object, [=](nd_item<1> item, auto& temp) {
       int i = item.get_global_id(0);
       temp.combine({data[i], i});
  }).wait();

  std::cout << "Minimum value and index = " << result->val << " at " << result->idx << "\n";

  free(result, q);
  free(data, q);
  return 0;
}

