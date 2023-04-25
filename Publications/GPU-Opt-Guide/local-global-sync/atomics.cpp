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

// Summation of 256k 'one' values
constexpr size_t N = 1024 * 256;

// Number of repetitions
constexpr int repetitions = 10000;
// expected vlaue of sum
int sum_expected = N;

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

int reductionIntSerial(sycl::queue &q, std::vector<int> &data, int iter) {
  const size_t data_size = data.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  // initialize data on the device
  q.submit([&](auto &h) {
    sycl::accessor buf_acc(buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(data_size, [=](auto index) { buf_acc[index] = 1; });
  });
  q.wait();

  double elapsed = 0;
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(1, [=](auto index) {
        size_t glob_id = index[0];
        sum_acc[0] = 0;
      });
    });

    Timer timer;
    // reductionIntSerial main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(data_size, [=](auto index) {
        int glob_id = index[0];
        if (glob_id == 0) {
          int sum = 0;
          for (int i = 0; i < N; i++)
            sum += buf_acc[i];
          sum_acc[0] = sum;
        }
      });
    });
    // reductionIntSerial main end
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
    elapsed += timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ReductionIntSerial   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ReductionIntSerial Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end reductionIntSerial

int reductionIntBarrier(sycl::queue &q, std::vector<int> &data, int iter) {
  const size_t data_size = data.size();
  int sum = 0;

  int work_group_size = 512;
  int num_work_items = work_group_size;
  int num_work_groups = num_work_items / work_group_size;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  // initialize data on the device
  q.submit([&](auto &h) {
    sycl::accessor buf_acc(buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(data_size, [=](auto index) { buf_acc[index] = 1; });
  });

  double elapsed = 0;
  for (int i = 0; i < iter; i++) {
    // reductionIntBarrier main begin
    Timer timer;
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      sycl::local_accessor<int, 1> scratch(work_group_size, h);
      h.parallel_for(sycl::nd_range<1>{work_group_size, work_group_size},
                     [=](sycl::nd_item<1> item) {
                       size_t loc_id = item.get_local_id(0);
                       int sum = 0;
                       for (int i = loc_id; i < data_size; i += num_work_items)
                         sum += buf_acc[i];
                       scratch[loc_id] = sum;
                       for (int i = work_group_size / 2; i > 0; i >>= 1) {
                         item.barrier(sycl::access::fence_space::local_space);
                         if (loc_id < i)
                           scratch[loc_id] += scratch[loc_id + i];
                       }
                       if (loc_id == 0)
                         sum_acc[0] = scratch[0];
                     });
    });
    // reductionIntBarrier main end
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
    elapsed += timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ReductionIntBarrier   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ReductionIntBarrier Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end ReductioNIntBarrier

int reductionIntAtomic(sycl::queue &q, std::vector<int> &data, int iter) {
  const size_t data_size = data.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  // initialize data on the device
  q.submit([&](auto &h) {
    sycl::accessor buf_acc(buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(data_size, [=](auto index) { buf_acc[index] = 1; });
  });
  q.wait();

  double elapsed = 0;
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(1, [=](auto index) { sum_acc[index] = 0; });
    });
    Timer timer;
    // reductionIntAtomic main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
    });
    // reductionIntAtomic main end
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
    elapsed += timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ReductionIntAtomic   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ReductionIntAtomic Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end reductionIntAtomic

int main(int argc, char *argv[]) {

  sycl::queue q{sycl::default_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
  std::vector<int> data(N, 1);
  reductionIntSerial(q, data, 1000);
  reductionIntAtomic(q, data, 1000);
  reductionIntBarrier(q, data, 1000);
}
