//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 256;

int main() {
  queue q;
 
  //# 3 vectors initialized to values
  std::vector<int> data1(N, 1);
  std::vector<int> data2(N, 2);
  std::vector<int> data3(N, 3);
 
  //# STEP 1 : Create buffers for data1, data2 and data3
  buffer data1_buf(data1);
  buffer data2_buf(data2);
  buffer data3_buf(data3);

  //# STEP 2 : Create a kernel to update data1 += data3, set accessor permissions
  q.submit([&](handler &h) {
    accessor a{data1_buf, h};
    accessor b{data3_buf, h, read_only};
    h.parallel_for(N, [=](auto i) { a[i] += b[i]; });
  });

  //# STEP 3 : Create a kernel to update data2 *= 2, set accessor permissions
  q.submit([&](handler &h) {
    accessor a{data2_buf, h};
    h.parallel_for(N, [=](auto i) { a[i] *= 2; });
  });

  //# STEP 4 : Create a kernel to update data3 = data1 + data2, set accessor permissions
  q.submit([&](handler &h) {
    accessor a{data3_buf, h, write_only};
    accessor b{data1_buf, h, read_only};
    accessor c{data2_buf, h, read_only};
    h.parallel_for(N, [=](auto i) { a[i] = b[i] + c[i]; });
  });

  //# STEP 5 : Create a host accessor to copy back data3
  host_accessor h_a{data3_buf};

  std::cout << "Output = " << data3[0] << "\n";
  return 0;
}
