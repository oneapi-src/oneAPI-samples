//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

int main() {
  constexpr int num_items = 1024*1000*1000;
  std::vector<int> a(num_items);
  for(int i=0;i<num_items;i++) a[i] = i;
  std::vector<int> b(num_items, 1);
  std::vector<int> c(num_items, 2);
  std::vector<int> d(num_items, 3);
  std::vector<int> sum(num_items, 0);
  std::vector<int> res(num_items, 0);

  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer c_buf(c, props);
  sycl::buffer d_buf(d, props);
  sycl::buffer sum_buf(sum, props);
  sycl::buffer res_buf(res, props);

  auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();

  //# Kernel 1
  q.submit([&](auto &h) {
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(num_items, [=](auto i) { sum_acc[i] = a_acc[i] + b_acc[i]; });
  });

  {
    sycl::host_accessor h_acc(sum_buf);
    for (int j = 0; j < num_items; j++)
      if (h_acc[j] > 10)
        h_acc[j] = 1;
      else
        h_acc[j] = 0;
  }

  //# Kernel 2
  q.submit([&](auto &h) {
    sycl::accessor c_acc(c_buf, h, sycl::read_only);
    sycl::accessor d_acc(d_buf, h, sycl::read_only);
    sycl::accessor sum_acc(sum_buf, h, sycl::read_only);
    sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);

    h.parallel_for(num_items, [=](auto i) { res_acc[i] = sum_acc[i] * c_acc[i] + d_acc[i]; });
  }).wait();

  sycl::host_accessor h_acc(res_buf); 
  for (int i = 0; i < 20; i++) std::cout << h_acc[i] << " ";std::cout << "...\n";
    
  auto duration = std::chrono::high_resolution_clock::now().time_since_epoch().count() - start;
  std::cout << "Compute Duration: " << duration / 1e+9 << " seconds\n";
  return 0;
}
