//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

// Summation of 10M 'one' values
constexpr size_t N = (10 * 1024 * 1024);

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

static auto exception_handler = [](sycl::exception_list eList) {
  for (std::exception_ptr const &e : eList) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

class Timer {
public:
  Timer() : start_(std::chrono::steady_clock::now()) {}

  double Elapsed() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start_).count();
  }

private:
  using Duration = std::chrono::duration<double>;
  std::chrono::steady_clock::time_point start_;
};

void reduction(sycl::queue &q, std::vector<int> &data, std::vector<int> &flush,
               int iter, int work_group_size) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  // int vec_size =
  // q.get_device().get_info<sycl::info::device::native_vector_width_int>();
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

    Timer timer;
    // reductionMapToHWVector main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(
          sycl::nd_range<1>(num_work_items, work_group_size),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            auto v =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    sum_acc[0]);
            int sum = 0;
            int glob_id = item.get_global_id();
            int loc_id = item.get_local_id();
            for (unsigned int i = glob_id; i < data_size; i += num_work_items)
              sum += buf_acc[i];
            scratch[loc_id] = sum;

            for (int i = work_group_size / 2; i > 0; i >>= 1) {
              item.barrier(sycl::access::fence_space::local_space);
              if (loc_id < i)
                scratch[loc_id] += scratch[loc_id + i];
            }

            if (loc_id == 0)
              v.fetch_add(scratch[0]);
          });
    });
    q.wait();
    elapsed += timer.Elapsed();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
  }
  elapsed = elapsed / iter;
  std::string msg = "with work-groups=" + std::to_string(work_group_size);
  check_result(elapsed, msg, sum);
} // reduction end

int main(void) {

  sycl::queue q{sycl::gpu_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<int> data(N, 1);
  std::vector<int> extra(N, 1);
  // call begin
  int vec_size = 16;
  int work_group_size = vec_size;
  reduction(q, data, extra, 16, work_group_size);
  work_group_size =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();
  reduction(q, data, extra, 16, work_group_size);
  // call end
}
