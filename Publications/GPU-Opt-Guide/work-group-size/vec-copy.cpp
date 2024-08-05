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

// Copy of 32M 'one' values
constexpr size_t N = (32 * 1024 * 1024);
typedef unsigned int uint;

void check_result(double elapsed, std::string msg, std::vector<int> &res) {
  bool ok = true;
  for (uint i = 0; i < N; i++) {
    if (res[i] != 1) {
      ok = false;
      std::cout << "ERROR: Mismatch at " << i << "\n";
    }
  }
  if (ok)
    std::cout << "SUCCESS: Time " << msg << " = " << elapsed << "s\n";
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

void vec_copy(sycl::queue &q, std::vector<int> &src, std::vector<int> &dst,
              std::vector<int> &flush, int iter, int work_group_size) {
  const size_t data_size = src.size();
  const size_t flush_size = flush.size();

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  int num_work_items = data_size;
  double elapsed = 0;
  {
    sycl::buffer<int> src_buf(src.data(), data_size, props);
    sycl::buffer<int> dst_buf(dst.data(), data_size, props);
    sycl::buffer<int> flush_buf(flush.data(), flush_size, props);

    for (int i = 0; i < iter; i++) {
      // flush the cache
      q.submit([&](auto &h) {
        sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
        h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
      });

      Timer timer;
      q.submit([&](auto &h) {
        sycl::accessor src_acc(src_buf, h, sycl::read_only);
        sycl::accessor dst_acc(dst_buf, h, sycl::write_only, sycl::no_init);

        h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
                       [=](sycl::nd_item<1> item)
                           [[intel::reqd_sub_group_size(16)]] {
                             int glob_id = item.get_global_id();
                             dst_acc[glob_id] = src_acc[glob_id];
                           });
      });
      q.wait();
      elapsed += timer.Elapsed();
    }
  }
  elapsed = elapsed / iter;
  std::string msg = "with work-group-size=" + std::to_string(work_group_size);
  check_result(elapsed, msg, dst);
} // vec_copy end

int main(void) {

  sycl::queue q{sycl::gpu_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<int> src(N, 1);
  std::vector<int> dst(N, 0);
  std::vector<int> extra(N, 1);

  // call begin
  int vec_size = 16;
  int work_group_size = vec_size;
  vec_copy(q, src, dst, extra, 16, work_group_size);
  work_group_size = 2 * vec_size;
  vec_copy(q, src, dst, extra, 16, work_group_size);
  work_group_size = 4 * vec_size;
  vec_copy(q, src, dst, extra, 16, work_group_size);
  work_group_size = 8 * vec_size;
  vec_copy(q, src, dst, extra, 16, work_group_size);
  work_group_size = 16 * vec_size;
  vec_copy(q, src, dst, extra, 16, work_group_size);
  // call end
}
