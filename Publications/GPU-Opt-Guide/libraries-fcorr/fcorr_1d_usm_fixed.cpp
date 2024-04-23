//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// Snippet begin
#include <CL/sycl.hpp>
#include <iostream>
#include <mkl.h>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>

int main(int argc, char **argv) {
  unsigned int N = (argc == 1) ? 32 : std::stoi(argv[1]);
  if ((N % 2) != 0)
    N++;
  if (N < 32)
    N = 32;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  auto sycl_device = Q.get_device();
  auto sycl_context = Q.get_context();
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Initialize signal and correlation arrays
  auto sig1 = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);
  auto sig2 = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);
  auto corr = sycl::malloc_shared<float>(N + 2, sycl_device, sycl_context);

  // Initialize input signals with artificial data
  std::uint32_t seed = (unsigned)time(NULL); // Get RNG seed value
  oneapi::mkl::rng::mcg31m1 engine(Q, seed); // Initialize RNG engine
                                             // Set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(-0.00005, 0.00005);

  auto evt1 =
      oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1); // Noise
  auto evt2 = oneapi::mkl::rng::generate(rng_distribution, engine, N, sig2);
  evt1.wait();
  evt2.wait();

  Q.single_task<>([=]() {
     sig1[N - N / 4 - 1] = 1.0;
     sig1[N - N / 4] = 1.0;
     sig1[N - N / 4 + 1] = 1.0; // Signal
     sig2[N / 4 - 1] = 1.0;
     sig2[N / 4] = 1.0;
     sig2[N / 4 + 1] = 1.0;
   }).wait();

  clock_t start_time = clock(); // Start timer

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
      Q, N / 2, reinterpret_cast<std::complex<float> *>(sig1),
      reinterpret_cast<std::complex<float> *>(sig2),
      reinterpret_cast<std::complex<float> *>(corr), {evt1, evt2})
      .wait();

  // Perform backward transform on complex correlation array
  oneapi::mkl::dft::compute_backward(transform_plan, corr).wait();

  clock_t end_time = clock(); // Stop timer
  std::cout << "The 1D correlation (N = " << N << ") took "
            << float(end_time - start_time) / CLOCKS_PER_SEC << " seconds."
            << std::endl;

  // Find the shift that gives maximum correlation value
  float max_corr = 0.0;
  int shift = 0;
  for (unsigned int idx = 0; idx < N; idx++) {
    if (corr[idx] > max_corr) {
      max_corr = corr[idx];
      shift = idx;
    }
  }
  int _N = static_cast<int>(N);
  shift =
      (shift > _N / 2) ? shift - _N : shift; // Treat the signals as circularly
                                             // shifted versions of each other.
  std::cout << "Shift the second signal " << shift
            << " elements relative to the first signal to get a maximum, "
               "normalized correlation score of "
            << max_corr / N << "." << std::endl;

  // Cleanup
  sycl::free(sig1, sycl_context);
  sycl::free(sig2, sycl_context);
  sycl::free(corr, sycl_context);
}
// Snippet end
