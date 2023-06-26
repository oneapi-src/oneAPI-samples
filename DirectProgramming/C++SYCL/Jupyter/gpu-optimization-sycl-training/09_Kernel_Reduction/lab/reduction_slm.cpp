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
  int sum = 0;

  int work_group_size = 256;
  int log2elements_per_block = 13;
  int elements_per_block = (1 << log2elements_per_block); // 8192

  int log2workitems_per_block = 8;
  int workitems_per_block = (1 << log2workitems_per_block); // 256
  int elements_per_work_item = elements_per_block / workitems_per_block;

  int mask = ~(~0 << log2workitems_per_block);
  int num_work_items = data.size() / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;
  std::cout << "Num work items = " << num_work_items << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> sum_buf(&sum, 1, props);
  sycl::buffer<int> accum_buf(num_work_groups);
    
  auto e = q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      h.parallel_for(sycl::nd_range<1>{num_work_items, work_group_size},
                     [=](sycl::nd_item<1> item) {
                       size_t glob_id = item.get_global_id(0);
                       size_t group_id = item.get_group(0);
                       size_t loc_id = item.get_local_id(0);
                       int offset = ((glob_id >> log2workitems_per_block)
                                     << log2elements_per_block) +
                                    (glob_id & mask);
                       int sum = 0;
                       for (size_t i = 0; i < elements_per_work_item; i++)
                         sum +=
                             buf_acc[(i << log2workitems_per_block) + offset];
                       scratch[loc_id] = sum;
                       // Serial Reduction
		       sycl::group_barrier(item.get_group());
                       if (loc_id == 0) {
                         int sum = 0;
                         for (int i = 0; i < work_group_size; i++)
                           sum += scratch[i];
                         accum_acc[group_id] = sum;
                       }
                     });
    });

    q.wait();
    {
      sum = 0;
      sycl::host_accessor h_acc(accum_buf);
      for (int i = 0; i < num_work_groups; i++)
        sum += h_acc[i];
    }
  std::cout << "Sum = " << sum << "\n";

  std::cout << "Kernel time = " << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9 << " seconds\n";
  return 0;
}
