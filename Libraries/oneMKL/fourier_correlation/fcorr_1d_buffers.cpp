//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the 1D Fourier correlation algorithm
//     using SYCL, oneMKL, oneDPL, and explicit buffering.
//
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>
#include <mkl.h>

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  unsigned int N = (argc == 1) ? 32 : std::stoi(argv[1]);
  if ((N % 2) != 0) N++;
  if (N < 32) N = 32;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Create buffers for signal data. This will only be used on the device.
  sycl::buffer<float> sig1_buf{N + 2};
  sycl::buffer<float> sig2_buf{N + 2};
  sycl::buffer<float> corr_buf{N + 2};

  // Initialize the input signals with artificial data
  std::uint32_t seed = (unsigned)time(NULL);  // Get RNG seed value
  oneapi::mkl::rng::mcg31m1 engine(Q, seed);  // Initialize RNG engine
                                              // Set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(-0.00005, 0.00005);

  oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1_buf);  // Noise
  oneapi::mkl::rng::generate(rng_distribution, engine, N, sig2_buf);

  Q.submit([&](sycl::handler &h) {
    sycl::accessor sig1_acc{sig1_buf, h, sycl::write_only};
    sycl::accessor sig2_acc{sig2_buf, h, sycl::write_only};
    h.single_task<>([=]() {
      sig1_acc[N - N / 4 - 1] = 1.0;
      sig1_acc[N - N / 4] = 1.0;
      sig1_acc[N - N / 4 + 1] = 1.0;  // Signal
      sig2_acc[N / 4 - 1] = 1.0;
      sig2_acc[N / 4] = 1.0;
      sig2_acc[N / 4 + 1] = 1.0;
    });
  });  // End signal initialization

  // Initialize FFT descriptor
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL>
      transform_plan(N);
  transform_plan.commit(Q);

  // Perform forward transforms on real arrays
  oneapi::mkl::dft::compute_forward(transform_plan, sig1_buf);
  oneapi::mkl::dft::compute_forward(transform_plan, sig2_buf);

  // Compute: DFT(sig1) * CONJG(DFT(sig2))
  auto sig1_buf_cplx =
      sig1_buf.template reinterpret<std::complex<float>, 1>((N + 2) / 2);
  auto sig2_buf_cplx =
      sig2_buf.template reinterpret<std::complex<float>, 1>((N + 2) / 2);
  auto corr_buf_cplx =
      corr_buf.template reinterpret<std::complex<float>, 1>((N + 2) / 2);
  oneapi::mkl::vm::mulbyconj(Q, N / 2, sig1_buf_cplx, sig2_buf_cplx,
                             corr_buf_cplx);

  // Perform backward transform on complex correlation array
  oneapi::mkl::dft::compute_backward(transform_plan, corr_buf);

  // Find the shift that gives maximum correlation value
  auto policy = oneapi::dpl::execution::make_device_policy(Q);
  auto maxloc = oneapi::dpl::max_element(policy,
                                         oneapi::dpl::begin(corr_buf),
                                         oneapi::dpl::end(corr_buf));
  int shift = oneapi::dpl::distance(oneapi::dpl::begin(corr_buf), maxloc);
  float max_corr = corr_buf.get_host_access()[shift];

  shift =
      (shift > N / 2) ? shift - N : shift;  // Treat the signals as circularly
                                            // shifted versions of each other.
  std::cout << "Shift the second signal " << shift
            << " elements relative to the first signal to get a maximum, "
               "normalized correlation score of "
            << max_corr / N << ".\n";
}
