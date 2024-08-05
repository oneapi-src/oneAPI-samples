//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>

int main() {
  constexpr int num_items = 16;
  constexpr int iter = 1;

  std::vector<int> a(num_items, 10);
  std::vector<int> b(num_items, 10);
  std::vector<int> sum(num_items, 0);

  sycl::queue q;
  std::cout << "Device : " << q.get_device().get_info<sycl::info::device::name>() << "\n";
  
  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer a_buf(a, props);
  sycl::buffer b_buf(b, props);
  sycl::buffer sum_buf(sum, props);
  {
    sycl::host_accessor a_host_acc(a_buf);
    std::cout << "address of vector a     = " << a.data() << "\n";
    std::cout << "buffer memory address   = " << a_host_acc.get_pointer() << "\n";
  }
  q.submit([&](auto &h) {
    // Input accessors
    sycl::accessor a_acc(a_buf, h, sycl::read_only);
    sycl::accessor b_acc(b_buf, h, sycl::read_only);
    // Output accessor
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    sycl::stream out(1024 * 1024, 1 * 128, h);

    h.parallel_for(num_items, [=](auto i) {
      if (i[0] == 0)
        out << "device accessor address = " << a_acc.get_pointer() << "\n";
      sum_acc[i] = a_acc[i] + b_acc[i];
    });
  }).wait();
  return 0;
}
