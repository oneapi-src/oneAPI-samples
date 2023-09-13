//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

constexpr size_t N = (1000 * 1024 * 1024);

int main(int argc, char *argv[]) {

  sycl::queue q{sycl::property::queue::enable_profiling{}};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<int> data(N, 1);
  int sum;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> sum_buf(&sum, 1, props);
    
  auto e = q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      auto sum_reduction = sycl::reduction(sum_buf, h, sycl::plus<>());
      h.parallel_for(sycl::nd_range<1>{N, 256}, sum_reduction,
                     [=](sycl::nd_item<1> item, auto &sum_wg) {
                       int i = item.get_global_id(0);
                       sum_wg += buf_acc[i];
                     });
    });

  sycl::host_accessor h_acc(sum_buf);
  std::cout << "Sum = " << sum << "\n";

  std::cout << "Kernel time = " << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9 << " seconds\n";
  return 0;
}
