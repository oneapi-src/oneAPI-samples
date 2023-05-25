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
  int log2elements_per_work_item = 6;
  int elements_per_work_item = (1 << log2elements_per_work_item); // 256
  int num_work_items = data.size() / elements_per_work_item;
  int num_work_groups = num_work_items / work_group_size;

  std::cout << "Num work items = " << num_work_items << std::endl;
  std::cout << "Num work groups = " << num_work_groups << std::endl;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data.size(), props);
  sycl::buffer<int> sum_buf(&sum, 1, props);
  sycl::buffer<sycl::vec<int, 8>> accum_buf(num_work_groups);
    
  auto e = q.submit([&](auto &h) {
      const sycl::accessor buf_acc(buf, h);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<sycl::vec<int, 8>, 1> scratch(work_group_size, h);
      h.parallel_for(
          sycl::nd_range<1>{num_work_items, work_group_size}, [=
      ](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            size_t glob_id = item.get_global_id(0);
            size_t group_id = item.get_group(0);
            size_t loc_id = item.get_local_id(0);
            sycl::ext::oneapi::sub_group sg = item.get_sub_group();
            sycl::vec<int, 8> sum{0, 0, 0, 0, 0, 0, 0, 0};
            using global_ptr =
                sycl::multi_ptr<int, sycl::access::address_space::global_space>;
            int base = (group_id * work_group_size +
                        sg.get_group_id()[0] * sg.get_local_range()[0]) *
                       elements_per_work_item;
            for (size_t i = 0; i < elements_per_work_item / 8; i++)
              sum += sg.load<8>(global_ptr(&buf_acc[base + i * 128]));
            scratch[loc_id] = sum;
            for (int i = work_group_size / 2; i > 0; i >>= 1) {
	    sycl::group_barrier(item.get_group());
              if (loc_id < i)
                scratch[loc_id] += scratch[loc_id + i];
            }
            if (loc_id == 0)
              accum_acc[group_id] = scratch[0];
          });
    });

    q.wait();
    {
      sycl::host_accessor h_acc(accum_buf);
      sycl::vec<int, 8> res{0, 0, 0, 0, 0, 0, 0, 0};
      for (int i = 0; i < num_work_groups; i++)
        res += h_acc[i];
      sum = 0;
      for (int i = 0; i < 8; i++)
        sum += res[i];
    }
  sycl::host_accessor h_acc(sum_buf);
  std::cout << "Sum = " << sum << "\n";

  std::cout << "Kernel time = " << (e.template get_profiling_info<sycl::info::event_profiling::command_end>() - e.template get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9 << " seconds\n";
  return 0;
}
