//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>

// Summation of 10M 'one' values
constexpr size_t N = (10 * 1024 * 1024);

// Number of repetitions
constexpr int repetitions = 16;
// expected vlaue of sum
int sum_expected = N;

void init_data(sycl::queue &q, sycl::buffer<int> &buf, int data_size) {
  // initialize data on the device
  q.submit([&](auto &h) {
    sycl::accessor buf_acc(buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(data_size, [=](auto index) { buf_acc[index] = 1; });
  });
  q.wait();
}

void check_result(double elapsed, std::string msg, int sum) {
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time is " << elapsed << "s" << msg << "\n";
  else
    std::cout << "ERROR: Expected " << sum_expected << " but got " << sum
              << "\n";
}

void reduction(sycl::queue &q, std::vector<int> &data, std::vector<int> &flush,
               int iter, int vec_size, int work_group_size) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_work_items = data_size / work_group_size;
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  init_data(q, buf, data_size);

  double elapsed = 0;
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(1, [=](auto index) { sum_acc[index] = 0; });
    });
    // flush the cache
    q.submit([&](auto &h) {
      sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
    });

    auto start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    // reductionMapToHWVector main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(
          sycl::nd_range<1>(num_work_items, work_group_size), [=
      ](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            auto v = sycl::atomic_ref<
                int, sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space>(sum_acc[0]);
            int sum = 0;
            int glob_id = item.get_global_id();
            int loc_id = item.get_local_id();
            for (int i = glob_id; i < data_size; i += num_work_items)
              sum += buf_acc[i];
            scratch[loc_id] = sum;

            for (int i = work_group_size / 2; i > 0; i >>= 1) {
	    sycl::group_barrier(item.get_group());
              if (loc_id < i)
                scratch[loc_id] += scratch[loc_id + i];
            }

            if (loc_id == 0)
              v.fetch_add(scratch[0]);
          });
    });
    q.wait();
    elapsed += (std::chrono::high_resolution_clock::now().time_since_epoch().count() - start) / 1e+9;
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
  }
  elapsed = elapsed / iter;
  std::string msg = " with work-groups=" + std::to_string(work_group_size);
  check_result(elapsed, msg, sum);
}

int main(int argc, char *argv[]) {

  sycl::queue q;
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<int> data(N, 1);
  std::vector<int> extra(N, 1);

  int vec_size = 16;
  int work_group_size = vec_size;
  reduction(q, data, extra, 16, vec_size, work_group_size);
  work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  reduction(q, data, extra, 16, vec_size, work_group_size);

}
