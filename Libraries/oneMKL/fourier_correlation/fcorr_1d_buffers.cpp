//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the Fourier correlation algorithm
//     using oneMKL, DPC++, and explicit buffering.
//
// =============================================================

#include <mkl.h>
#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>

int main(int argc, char **argv) {
  unsigned int N = (argc == 1) ? 32 : std::stoi(argv[1]);
  if ((N % 2) != 0) N++;
  if (N < 32) N = 32;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector{});
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Create buffers for signal data. This will only be used on the device.
  sycl::buffer<float> sig1_buf{N + 2};
  sycl::buffer<float> sig2_buf{N + 2};

  // Declare container to hold the correlation result (computed on the device,
  // used on the host)
  std::vector<float> corr(N + 2);

  // Open new scope to trigger update of correlation result
  {
    sycl::buffer<float> corr_buf(corr);

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

  }  // Buffer holding correlation result is now out of scope, forcing update of
     // host container

  // Find the shift that gives maximum correlation value
  float max_corr = 0.0;
  int shift = 0;
  for (unsigned int idx = 0; idx < N; idx++) {
    if (corr[idx] > max_corr) {
      max_corr = corr[idx];
      shift = idx;
    }
  }
  shift =
      (shift > N / 2) ? shift - N : shift;  // Treat the signals as circularly
                                            // shifted versions of each other.
  std::cout << "Shift the second signal " << shift
            << " elements relative to the first signal to get a maximum, "
               "normalized correlation score of "
            << max_corr / N << ".\n";
}
