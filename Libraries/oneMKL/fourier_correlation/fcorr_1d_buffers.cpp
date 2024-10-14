//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the 1D Fourier correlation algorithm
//     using SYCL, oneMKL, and explicit buffering.
//
// =============================================================

#include <mkl.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <oneapi/mkl/dft.hpp>
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/vm.hpp>
#include <oneapi/mkl/blas.hpp>

static void
naive_cross_correlation(sycl::queue& Q,
                        unsigned int N,
                        sycl::buffer<float>& u,
                        sycl::buffer<float>& v,
                        sycl::buffer<float>& w) {
  const size_t min_byte_size = N * sizeof(float);
  if (u.byte_size() < min_byte_size ||
      v.byte_size() < min_byte_size ||
      w.byte_size() < min_byte_size) {
    throw std::invalid_argument("All buffers must contain at least N float values");
  }
  Q.submit([&](sycl::handler &cgh) {
    auto u_acc = u.get_access<sycl::access::mode::read>(cgh);
    auto v_acc = v.get_access<sycl::access::mode::read>(cgh);
    auto w_acc = w.get_access<sycl::access::mode::write>(cgh);
    cgh.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> id) {
      const size_t s = id.get(0);
      w_acc[s] = 0.0f;
      for (size_t j = 0; j < N; j++) {
        w_acc[s] += u_acc[j] * v_acc[(j - s + N) % N];
      }
    });
  });
}

int main(int argc, char **argv) {
  unsigned int N = (argc == 1) ? 32 : std::stoi(argv[1]);
  // N >= 8 required for the arbitrary signals as defined herein
  if (N < 8)
    throw std::invalid_argument("The input value N must be 8 or greater.");

  // Let s be an integer s.t. 0 <= s < N and let
  //       corr[s] = \sum_{j = 0}^{N-1} sig1[j] sig2[(j - s + N) mod N]
  // be the cross-correlation between two real periodic signals sig1 and sig2
  // of period N. This code shows how to calculate corr using Discrete Fourier
  // Transforms (DFTs).
  // 0 (resp. 1) is returned if naive and DFT-based calculations are (resp.
  // are not) within error tolerance of one another.
  int return_code = 0;

  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;
  // Initialize signal and correlation buffers. The buffers must be large enough
  // to store the forward and backward domains' data, consisting of N real
  // values and (N/2 + 1) complex values, respectively (for the DFT-based
  // calculations).
  sycl::buffer<float> sig1{2 * (N / 2 + 1)};
  sycl::buffer<float> sig2{2 * (N / 2 + 1)};
  sycl::buffer<float> corr{2 * (N / 2 + 1)};
  // Buffer used for calculating corr without Discrete Fourier Transforms
  // (for comparison purposes):
  sycl::buffer<float> naive_corr{N};

  // Initialize input signals with artificial "noise" data (random values of
  // magnitude much smaller than relevant signal data points)
  std::uint32_t seed = (unsigned)time(NULL);  // Get RNG seed value
  oneapi::mkl::rng::mcg31m1 engine(Q, seed);  // Initialize RNG engine
                                              // Set RNG distribution
  oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard>
      rng_distribution(-0.00005f, 0.00005f);

  oneapi::mkl::rng::generate(rng_distribution, engine, N, sig1);
  oneapi::mkl::rng::generate(rng_distribution, engine, N, sig2);

  // Set the (relevant) signal data as shifted versions of one another
  Q.submit([&](sycl::handler &cgh) {
    sycl::accessor sig1_acc{sig1, cgh, sycl::write_only};
    sycl::accessor sig2_acc{sig2, cgh, sycl::write_only};
    cgh.single_task<>([=]() {
      sig1_acc[N - N / 4 - 1] = 1.0f;
      sig1_acc[N - N / 4]     = 1.0f;
      sig1_acc[N - N / 4 + 1] = 1.0f;
      sig2_acc[N / 4 - 1]     = 1.0f;
      sig2_acc[N / 4]         = 1.0f;
      sig2_acc[N / 4 + 1]     = 1.0f;
    });
  });
  // Calculate L2 norms of both input signals before proceeding (for
  // normalization purposes and for the definition of error tolerance)
  float norm_sig1, norm_sig2;
  {
    sycl::buffer<float> temp{1};
    oneapi::mkl::blas::nrm2(Q, N, sig1, 1, temp);
    norm_sig1 = temp.get_host_access(sycl::read_only)[0];
    oneapi::mkl::blas::nrm2(Q, N, sig2, 1, temp);
    norm_sig2 = temp.get_host_access(sycl::read_only)[0];
  }
  // 1) Calculate the cross-correlation naively (for verification purposes);
  naive_cross_correlation(Q, N, sig1, sig2, naive_corr);
  // 2) Calculate the cross-correlation via Discrete Fourier Transforms (DFTs):
  //       corr = (1/N) * iDFT(DFT(sig1) * CONJ(DFT(sig2)))
  // Initialize DFT descriptor
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL> desc(N);
  desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, 1.0f / N);
  desc.commit(Q);
  // Compute in-place forward transforms of both signals:
  // sig1 <- DFT(sig1)
  oneapi::mkl::dft::compute_forward(desc, sig1);
  // sig2 <- DFT(sig2)
  oneapi::mkl::dft::compute_forward(desc, sig2);
  // Compute the element-wise multipication of (complex) coefficients in
  // backward domain:
  // corr <- sig1 * CONJ(sig2) [component-wise]
  auto sig1_cplx =
      sig1.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
  auto sig2_cplx =
      sig2.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
  auto corr_cplx =
      corr.template reinterpret<std::complex<float>, 1>(N / 2 + 1);
  oneapi::mkl::vm::mulbyconj(Q, N / 2 + 1,
                             sig1_cplx, sig2_cplx, corr_cplx);
  // Compute in-place (scaled) backward transform:
  // corr <- (1/N) * iDFT(corr)
  oneapi::mkl::dft::compute_backward(desc, corr);

  // Error bound for naive calculations:
  float max_err_threshold =
    2.0f * std::numeric_limits<float>::epsilon() * norm_sig1 * norm_sig2;
  // Adding an (empirical) error bound for the DFT-based calculation defined as
  //           epsilon * O(log(N)) * scaling_factor * nrm2(input data),
  // wherein (for the last DFT at play)
  // - scaling_factor = 1.0 / N;
  // - nrm2(input data) = norm_sig1 * norm_sig2 * N
  // - O(log(N)) ~ 2 * log(N) [arbitrary choice; implementation-dependent behavior]
  max_err_threshold +=
    2.0f * std::log(static_cast<float>(N))
         * std::numeric_limits<float>::epsilon() * norm_sig1 * norm_sig2;
  // Verify results by comparing DFT-based and naive calculations to each other,
  // and fetch optimal shift maximizing correlation (DFT-based calculation).
  auto naive_corr_acc = naive_corr.get_host_access(sycl::read_only);
  auto corr_acc = corr.get_host_access(sycl::read_only);
  float max_err = 0.0f;
  float max_corr = corr_acc[0];
  int optimal_shift = 0;
  for (size_t s = 0; s < N; s++) {
    const float local_err = fabs(naive_corr_acc[s] - corr_acc[s]);
    if (local_err > max_err)
      max_err = local_err;
    if (max_err > max_err_threshold) {
      std::cerr << "An error was found when verifying the results." << std::endl;
      std::cerr << "For shift value s = " << s << ":" << std::endl;
      std::cerr << "\tNaive calculation results in " << naive_corr_acc[s] << std::endl;
      std::cerr << "\tFourier-based calculation results in " << corr_acc[s] << std::endl;
      std::cerr << "The error (" << max_err
                << ") exceeds the threshold value of "
                << max_err_threshold <<  std::endl;
      return_code = 1;
      break;
    }
    if (corr_acc[s] > max_corr) {
      max_corr = corr_acc[s];
      optimal_shift = s;
    }
  }
  // Conclude:
  if (return_code == 0) {
    // Get average and standard deviation of either signal for normalizing the
    // correlation "score"
    const float avg_sig1 = sig1.get_host_access(sycl::read_only)[0] / N;
    const float avg_sig2 = sig2.get_host_access(sycl::read_only)[0] / N;
    const float std_dev_sig1 =
          std::sqrt((norm_sig1 * norm_sig1 - N * avg_sig1 * avg_sig1) / N);
    const float std_dev_sig2 =
          std::sqrt((norm_sig2 * norm_sig2 - N * avg_sig2 * avg_sig2) / N);
    const float normalized_corr =
      (max_corr / N - avg_sig1 * avg_sig2) / (std_dev_sig1 * std_dev_sig2);
    std::cout << "Right-shift the second signal " << optimal_shift
              << " elements to get a maximum, normalized correlation score of "
              << normalized_corr
              << " (treating the signals as periodic)." << std::endl;
    std::cout << "Max difference between naive and Fourier-based calculations : "
              << max_err << " (verification threshold: " << max_err_threshold
              << ")." << std::endl;
  }
  return return_code;
}
