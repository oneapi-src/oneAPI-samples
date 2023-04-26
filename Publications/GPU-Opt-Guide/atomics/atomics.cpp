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
constexpr size_t N = 1024 * 32;

// Number of repetitions
constexpr int repetitions = 16;
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

int ComputeSerialInt(std::vector<int> &data, std::vector<int> &flush,
                     int iter) {
  const size_t data_size = data.size();
  Timer timer;
  int sum;
  // ComputeSerial main begin
  for (int it = 0; it < iter; it++) {
    sum = 0;
    for (size_t i = 0; i < data_size; ++i) {
      sum += data[i];
    }
  }
  // ComputeSerial main end
  double elapsed = timer.Elapsed() / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeSerialInt   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ComputeSerialInt Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end ComputeSerial

int ComputeSerialFloat(std::vector<float> &data, std::vector<float> &flush,
                       int iter) {
  const size_t data_size = data.size();
  Timer timer;
  float sum;
  // ComputeSerial main begin
  for (int it = 0; it < iter; it++) {
    sum = 0.0;
    for (size_t i = 0; i < data_size; ++i) {
      sum += data[i];
    }
  }
  // ComputeSerial main end
  double elapsed = timer.Elapsed() / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ComputeSerialFloat   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ComputeSerialFloat Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end ComputeSerial

int reductionInt(sycl::queue &q, std::vector<int> &data,
                 std::vector<int> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
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
    // flush the cache
    q.submit([&](auto &h) {
      sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
    });

    Timer timer;
    // reductionInt main begin
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
    // reductionInt main end
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
    elapsed += timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ReductionInt   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ReductionInt Expected " << sum_expected << " but got "
              << sum << "\n";
  return sum;
} // end reduction1

int reductionFloat(sycl::queue &q, std::vector<float> &data,
                   std::vector<float> &flush, int iter) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  float sum = 0.0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

  sycl::buffer<float> buf(data.data(), data_size, props);
  sycl::buffer<float> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<float> sum_buf(&sum, 1, props);

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
    // flush the cache
    q.submit([&](auto &h) {
      sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
    });

    Timer timer;
    // reductionFloat main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(data_size, [=](auto index) {
        size_t glob_id = index[0];
        auto v = sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum_acc[0]);
        v.fetch_add(buf_acc[glob_id]);
      });
    });
    // reductionFloat main end
    q.wait();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
    elapsed += timer.Elapsed();
  }
  elapsed = elapsed / iter;
  if (sum == sum_expected)
    std::cout << "SUCCESS: Time ReductionFLoat   = " << elapsed << "s"
              << " sum = " << sum << "\n";
  else
    std::cout << "ERROR: ReductionFloat Expected " << sum_expected
              << " but got " << sum << "\n";
  return sum;
} // end reduction2

int main(int argc, char *argv[]) {

  sycl::queue q{sycl::default_selector_v, exception_handler};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
  {
    std::vector<int> data(N, 1);
    std::vector<int> extra(N, 1);
    ComputeSerialInt(data, extra, 16);
    reductionInt(q, data, extra, 16);
  }

  {
    std::vector<float> data(N, 1.0f);
    std::vector<float> extra(N, 1.0f);
    ComputeSerialFloat(data, extra, 16);
    reductionFloat(q, data, extra, 16);
  }
}
