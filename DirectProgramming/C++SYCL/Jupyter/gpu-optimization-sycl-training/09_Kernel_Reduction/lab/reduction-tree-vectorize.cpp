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
  int num_processing_elements =
      q.get_device().get_info<sycl::info::device::max_compute_units>();
  int vec_size =
      q.get_device().get_info<sycl::info::device::native_vector_width_int>();
  int num_work_items = num_processing_elements * vec_size;

  std::cout << "Num work items = " << num_work_items << std::endl;

  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> accum_buf(num_work_items);
    
  auto e = q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_work_items, [=](auto index) {
        size_t glob_id = index[0];
        int sum = 0;
        for (size_t i = glob_id; i < N; i += num_work_items)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });

  sycl::host_accessor h_acc(accum_buf);
  for (int i = 0; i < num_work_items; i++) sum += h_acc[i];
  std::cout << "Sum = " << sum << "\n";

  std::cout << "Kernel time = " << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9 << " seconds\n";
  return 0;
}
