//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>

const int iters = 10000;
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

void emptyKernel1(sycl::queue &q) {
  Timer timer;
  for (int i = 0; i < iters; ++i)
    q.parallel_for(1, [=](auto) {
       /* NOP */
     }).wait();
  std::cout << "  emptyKernel1: Elapsed time: " << timer.Elapsed() / iters
            << " sec\n";
} // end emptyKernel1

void emptyKernel2(sycl::queue &q) {
  Timer timer;
  for (int i = 0; i < iters; ++i)
    q.parallel_for(1, [=](auto) {
      /* NOP */
    });
  std::cout << "  emptyKernel2: Elapsed time: " << timer.Elapsed() / iters
            << " sec\n";
} // end emptyKernel2

int main() {
  sycl::queue q;

  emptyKernel1(q);
  emptyKernel2(q);

  return 0;
}
