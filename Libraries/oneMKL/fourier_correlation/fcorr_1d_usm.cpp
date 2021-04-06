//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the Fourier correlation algorithm
//     using oneMKL, DPC++, and unified shared memory (USM).
//
// =============================================================

#include <mkl.h>
#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>

template <typename T, typename I>
struct pair {
  bool operator>(const pair& o) const {
    return val >= o.val || (val == o.val && idx >= o.idx);
  }
  T val;
  I idx;
};

template <typename T, typename I>
using maxloc = sycl::ONEAPI::maximum<pair<T, I>>;

int main(int argc, char** argv) {
  unsigned int N = (argc == 1) ? 32 : std::stoi(argv[1]);
  if ((N % 2) != 0) N++;
  if (N < 32) N = 32;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector{});
  auto sycl_device = Q.get_device();
  auto sycl_context = Q.get_context();
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Initialize signal and correlation arrays
  auto sig1 = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);
  auto sig2 = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);
  auto corr = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);

  // Initialize input signals with artificial data
  std::uint32_t seed = (unsigned)time(NULL);  // Get RNG seed value
  oneapi::mkl::rng::mcg31m1 engine(Q, seed);  // Initialize RNG engine
                                              // Set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(-0.00005, 0.00005);

  auto evt1 =
      oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1);  // Noise
  auto evt2 = oneapi::mkl::rng::generate(rng_distribution, engine, N, sig2);
  evt1.wait();
  evt2.wait();

  Q.single_task<>([=]() {
     sig1[N - N / 4 - 1] = 1.0;
     sig1[N - N / 4] = 1.0;
     sig1[N - N / 4 + 1] = 1.0;  // Signal
     sig2[N / 4 - 1] = 1.0;
     sig2[N / 4] = 1.0;
     sig2[N / 4 + 1] = 1.0;
   })
      .wait();

  // Initialize FFT descriptor
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL>
      transform_plan(N);
  transform_plan.commit(Q);

  // Perform forward transforms on real arrays
  evt1 = oneapi::mkl::dft::compute_forward(transform_plan, sig1);
  evt2 = oneapi::mkl::dft::compute_forward(transform_plan, sig2);

  // Compute: DFT(sig1) * CONJG(DFT(sig2))
  oneapi::mkl::vm::mulbyconj(
      Q, N / 2, reinterpret_cast<std::complex<float>*>(sig1),
      reinterpret_cast<std::complex<float>*>(sig2),
      reinterpret_cast<std::complex<float>*>(corr), {evt1, evt2})
      .wait();

  // Perform backward transform on complex correlation array
  oneapi::mkl::dft::compute_backward(transform_plan, corr).wait();

  // Find the shift that gives maximum correlation value using parallel maxloc
  // reduction
  pair<float, int>* max_res = sycl::malloc_shared<pair<float, int>>(1, Q);
  pair<float, int> max_identity = {std::numeric_limits<float>::min(),
                                   std::numeric_limits<int>::min()};
  *max_res = max_identity;
  auto red_max = reduction(max_res, max_identity, maxloc<float, int>());

  Q.submit([&](sycl::handler& h) {
     h.parallel_for(sycl::nd_range<1>{N, N / 16}, red_max,
                    [=](sycl::nd_item<1> item, auto& max_res) {
                      int i = item.get_global_id(0);
                      pair<float, int> partial = {corr[i], i};
                      max_res.combine(partial);
                    });
   })
      .wait();
  float max_corr = max_res->val;
  int shift = max_res->idx;
  shift =
      (shift > N / 2) ? shift - N : shift;  // Treat the signals as circularly
                                            // shifted versions of each other.
  std::cout << "Shift the second signal " << shift
            << " elements relative to the first signal to get a maximum, "
               "normalized correlation score of "
            << max_corr / N << ".\n";

  // Cleanup
  sycl::free(sig1, sycl_context);
  sycl::free(sig2, sycl_context);
  sycl::free(corr, sycl_context);
  sycl::free(max_res, sycl_context);
}
