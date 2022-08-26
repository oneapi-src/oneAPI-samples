//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// Summation of 10M 'one' values
constexpr size_t N = (10 * 1024 * 1024);

// Number of repetitions
constexpr int repetitions = 16;

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

float ComputeSerial(std::vector<float> &data) {
  const size_t data_size = data.size();
  float sum = 0;
  for (size_t i = 0; i < data_size; ++i) {
    sum += data[i];
  }
  return sum;
} // end ComputeSerial

float ComputeParallel1(sycl::queue &q, std::vector<float> &data) {
  const size_t data_size = data.size();
  float sum = 0;
  static float *accum = 0;

  if (data_size > 0) {
    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    int num_EUs =
        q.get_device().get_info<sycl::info::device::max_compute_units>();
    int vec_size =
        q.get_device()
            .get_info<sycl::info::device::native_vector_width_float>();
    int num_processing_elements = num_EUs * vec_size;
    int BATCH = (N + num_processing_elements - 1) / num_processing_elements;
    sycl::buffer<float> buf(data.data(), data.size(), props);
    sycl::buffer<float> accum_buf(accum, num_processing_elements, props);

    if (!accum)
      accum = new float[num_processing_elements];

    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_processing_elements, [=](auto index) {
        size_t glob_id = index[0];
        size_t start = glob_id * BATCH;
        size_t end = (glob_id + 1) * BATCH;
        if (end > N)
          end = N;
        float sum = 0.0;
        for (size_t i = start; i < end; i++)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });
    q.wait();
    sycl::host_accessor h_acc(accum_buf);
    for (int i = 0; i < num_processing_elements; i++)
      sum += h_acc[i];
  }
  return sum;
} // end ComputeParallel1

float ComputeParallel2(sycl::queue &q, std::vector<float> &data) {
  const size_t data_size = data.size();
  float sum = 0;
  static float *accum = 0;

  if (data_size > 0) {
    const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
    int num_EUs =
        q.get_device().get_info<sycl::info::device::max_compute_units>();
    int vec_size =
        q.get_device()
            .get_info<sycl::info::device::native_vector_width_float>();
    int num_processing_elements = num_EUs * vec_size;
    int BATCH = (N + num_processing_elements - 1) / num_processing_elements;
    sycl::buffer<float> buf(data.data(), data.size(), props);
    sycl::buffer<float> accum_buf(accum, num_processing_elements, props);
    sycl::buffer<float> res_buf(&sum, 1, props);
    if (!accum)
      accum = new float[num_processing_elements];

    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(num_processing_elements, [=](auto index) {
        size_t glob_id = index[0];
        size_t start = glob_id * BATCH;
        size_t end = (glob_id + 1) * BATCH;
        if (end > N)
          end = N;
        float sum = 0.0;
        for (size_t i = start; i < end; i++)
          sum += buf_acc[i];
        accum_acc[glob_id] = sum;
      });
    });

    q.submit([&](auto &h) {
      sycl::accessor accum_acc(accum, h, sycl::read_only);
      sycl::accessor res_acc(res_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(1, [=](auto index) {
        res_acc[index] = 0;
        for (size_t i = 0; i < num_processing_elements; i++)
          res_acc[index] += accum_acc[i];
      });
    });
  }
  // Buffers go out of scope and data gets transferred from device to host
  return sum;
} // end ComputeParallel2

int main(int argc, char *argv[]) {

  sycl::queue q{default_selector{}, exception_handler};
  std::vector<float> data(N, 1.0f);
  std::vector<float> extra(4 * N, 1.0f);

  float sum_s = 0, sum = 0;
  double elapsed_s = 0;
  for (int k = 0; k < repetitions; ++k) {
    // Flush the cache
    (void)ComputeSerial(extra);

    Timer timer_s;
    // Time the summation
    sum_s = ComputeSerial(data);
    elapsed_s += timer_s.Elapsed();
  }
  elapsed_s /= repetitions;

  std::cout << "Time Serial   = " << elapsed_s << "s"
            << " sum = " << sum_s << "\n";

  double elapsed_p1 = 0;
  for (int k = 0; k < repetitions; ++k) {
    // Flush the cache
    (void)ComputeParallel1(q, extra);

    Timer timer_s;
    // Time the summation
    sum = ComputeParallel1(q, data);
    elapsed_p1 += timer_s.Elapsed();
  }
  elapsed_p1 /= repetitions;
  std::cout << "Time parallel1   = " << elapsed_p1 << "s"
            << " sum = " << sum << "\n";

  double elapsed_p2 = 0;
  for (int k = 0; k < repetitions; ++k) {
    // Flush the cache
    (void)ComputeParallel2(q, extra);

    Timer timer_s;
    // Time the summation
    sum = ComputeParallel2(q, data);
    elapsed_p2 += timer_s.Elapsed();
  }
  elapsed_p2 /= repetitions;
  std::cout << "Time parallel2   = " << elapsed_p2 << "s"
            << " sum = " << sum << "\n";

  return 0;
}
