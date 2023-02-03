//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

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

int main() {
  Timer timer;
  sycl::queue q{sycl::property::queue::enable_profiling()};
  auto evt = q.parallel_for(1000, [=](auto) {
    /* kernel statements here */
  });
  double t1 = timer.Elapsed();
  evt.wait();
  double t2 = timer.Elapsed();
  auto startK =
      evt.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto endK =
      evt.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << "Kernel submission time: " << t1 << "secs\n";
  std::cout << "Kernel submission + execution time: " << t2 << "secs\n";
  std::cout << "Kernel execution time: "
            << ((double)(endK - startK)) / 1000000.0 << "secs\n";

  return 0;
}
