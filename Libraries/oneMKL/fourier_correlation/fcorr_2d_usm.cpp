//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the 2D Fourier correlation algorithm
//     using SYCL, oneMKL, and unified shared memory (USM).
//
// =============================================================

#include <mkl.h>
#include <sycl/sycl.hpp>
#include <iostream>
#include <oneapi/mkl/dft.hpp>
#include <oneapi/mkl/vm.hpp>

template <typename T>
static bool is_device_accessible(const T* x, const sycl::queue& Q) {
  sycl::usm::alloc alloc_type = sycl::get_pointer_type(x, Q.get_context());
  return (alloc_type == sycl::usm::alloc::shared
          || alloc_type == sycl::usm::alloc::device);
}

template <typename T>
static bool is_host_accessible(const T* x, const sycl::queue& Q) {
  sycl::usm::alloc alloc_type = sycl::get_pointer_type(x, Q.get_context());
  return (alloc_type == sycl::usm::alloc::shared
          || alloc_type == sycl::usm::alloc::host);
}

static void
print_image(const float* img, const sycl::queue& Q,
            const unsigned& n_rows, const unsigned& n_cols,
            const unsigned& col_stride, const std::string& header) {
  if (!is_host_accessible(img, Q)) {
    throw std::invalid_argument("img must be host-accessible");
  }
  std::cout << header << std::endl;
  // img must contain (at least)
  // ((n_rows - 1) * col_stride + n_cols - 1) float values
  for (auto row = 0; row < n_rows; row++) {
    for (auto col = 0; col < n_cols; col++) {
      std::cout << img[row * col_stride + col] << " ";
    }
    std::cout << std::endl;
  }
}

static float
calc_frobenius_norm(const float* img, sycl::queue& Q,
                    const unsigned& n_rows, const unsigned& n_cols,
                    const unsigned& col_stride) {
  if (!is_device_accessible(img, Q)) {
    throw std::invalid_argument("img must be device-accessible");
  }
  float* temp = sycl::malloc_shared<float>(1, Q);
  temp[0] = 0.0f;
  Q.submit([&](sycl::handler &cgh) {
    auto sumReduction = sycl::reduction(temp, 0.0f, sycl::plus<float>());
    cgh.parallel_for(sycl::range<2>{n_rows, n_cols},
                     sumReduction,
                     [=](sycl::id<2> idx, auto& sum) {
      size_t row = idx[0];
      size_t col = idx[1];
      sum += img[row * col_stride + col] * img[row * col_stride + col];
    });
  }).wait();
  const float frobenius_norm = std::sqrt(temp[0]);
  sycl::free(temp, Q);
  return frobenius_norm;
}

static sycl::event
naive_cross_correlation(sycl::queue& Q,
                        const unsigned& n_rows,
                        const unsigned& n_cols,
                        const unsigned& col_stride,
                        const float* u,
                        const float* v,
                        float* w,
                        const std::vector<sycl::event> deps = {}) {
  // u, v and w must be USM allocations of (at least)
  // ((n_rows - 1) * col_stride + n_cols - 1) float values (w must be writable)
  if (!is_device_accessible(u, Q) ||
      !is_device_accessible(v, Q) ||
      !is_device_accessible(w, Q)) {
    throw std::invalid_argument("Image arrays must be device-accessible");
  }
  sycl::event ev = Q.parallel_for(sycl::range<2>{n_rows, n_cols},
                                  [=](sycl::id<2> idx) {
    const size_t s = idx[0];
    const size_t p = idx[1];
    const size_t w_idx = s * col_stride + p;
    w[w_idx] = 0.0f;
    for (size_t j = 0; j < n_rows; j++) {
      for (size_t k = 0; k < n_cols; k++) {
        w[w_idx] += u[j * col_stride + k] *
                    v[((j - s + n_rows) % n_rows) * col_stride
                        + ((k - p + n_cols) % n_cols)];
      }
    }
  });
  return ev;
}

int main(int argc, char **argv) {
  int temp = (argc <= 1) ? 8 : std::stoi(argv[1]);
  // n_rows >= 6 required for the arbitrary signals as defined herein
  if (temp < 6)
    throw std::invalid_argument("The number of rows of the images, chosen as "
                          "first input of the program, must be 6 or greater.");
  const unsigned n_rows = temp;
  temp = (argc <= 2) ? 8 : std::stoi(argv[2]);
  // n_cols >= 7 required for the arbitrary signals as defined herein
  if (temp < 7)
    throw std::invalid_argument("The number of columns of the images, chosen as "
                          "second input of the program, must be 7 or greater.");
  const unsigned n_cols = temp;
  const unsigned num_elem = n_rows * n_cols;

  // Let s and p be integer s.t. 0 <= s < n_rows, 0 <= p < n_cols, and let
  // corr(s, p) =
  //  \sum_{j = 0}^{n_rows - 1} \sum_{k = 0}^{n_cols - 1} \
  //    img1(j, k) * img2((j - s + n_rows) mod n_rows, (k - p + n_cols) mod n_cols)
  // be the cross-correlation between two real periodic signals img1 and img2
  // of periods (n_rows, n_cols). This code shows how to calculate corr using
  // Discrete Fourier Transforms (DFTs).
  // Note: in the above, the notation "x(i, j)" represents the entry of
  // multi-index (i, j) within a (real) data sequence "x". If the latter is
  // stored in memory in an allocation "x_alloc" using a unit row stride and a
  // column stride col_stride (col_stride >= n_cols), we have
  //           x_alloc[i * col_stride + j] <- "x(i,j)"

  // This program returns 0 (resp. 1) if naive and DFT-based calculations are
  // (resp. are not) within error tolerance of one another.
  int return_code = 0;
  
  // Initialize SYCL queue
  sycl::queue Q(sycl::default_selector_v);
  std::cout << "Running on: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << std::endl;

  // Initialize 2D image and correlation arrays. The arrays must be large enough
  // to store the forward and backward domains' data, consisting of
  // n_rows * n_cols real values and n_rows * (n_cols / 2 + 1) complex values,
  // respectively (for the DFT-based calculations).
  // Note: 2 * (n_cols / 2 + 1) > n_cols for all n_cols > 0, so
  //    max(n_rows * n_cols real,
  //        n_rows * (n_cols / 2 + 1) * 2) == n_rows * (n_cols / 2 + 1) * 2
  // since n_rows > 0
  auto img1 = sycl::malloc_shared<float>(n_rows * (n_cols / 2 + 1) * 2, Q);
  auto img2 = sycl::malloc_shared<float>(n_rows * (n_cols / 2 + 1) * 2, Q);
  auto corr = sycl::malloc_shared<float>(n_rows * (n_cols / 2 + 1) * 2, Q);
  // For in-place calculations, the address of the 0th element in every row must
  // also be identical in forward and backward domains so we use
  const unsigned col_stride_fwd_domain = 2 * (n_cols / 2 + 1);
  // in forward (real) domain.
  // Note: col_stride_fwd_domain / 2 is to be used in backward (complex)
  // domain since complex values consist of 2 contiguous real values.

  // Initialize array for calculating corr without Discrete Fourier Transforms
  // (for comparison purposes). The same column stride as for DFT-based
  // calculations is used to satisfy the requirements of naive_cross_correlation:
  auto naive_corr = sycl::malloc_shared<float>(n_rows * (n_cols / 2 + 1) * 2, Q);

  // Set the relevant image data as shifted versions of one another
  auto evt = Q.parallel_for(sycl::range<2>{n_rows, n_cols},
                           [=](sycl::id<2> idx) {
    size_t row = idx[0];
    size_t col = idx[1];
    img1[row * col_stride_fwd_domain + col] = 0.0f;
    img2[row * col_stride_fwd_domain + col] = 0.0f;
  });
  Q.single_task<>({evt}, [=]() {
    // Set a box of unit elements in multi-indices (4-5, 5-6) for the first
    // image
    img1[4 * col_stride_fwd_domain + 5] = 1.0f;
    img1[4 * col_stride_fwd_domain + 6] = 1.0f;
    img1[5 * col_stride_fwd_domain + 5] = 1.0f;
    img1[5 * col_stride_fwd_domain + 6] = 1.0f;
    // Set a box of unit elements in multi-indices (1-2, 1-2) for the second
    // image
    img2[1 * col_stride_fwd_domain + 1] = 1.0f;
    img2[1 * col_stride_fwd_domain + 2] = 1.0f;
    img2[2 * col_stride_fwd_domain + 1] = 1.0f;
    img2[2 * col_stride_fwd_domain + 2] = 1.0f;
  }).wait();

  print_image(img1, Q, n_rows, n_cols, col_stride_fwd_domain, "First image:");
  print_image(img2, Q, n_rows, n_cols, col_stride_fwd_domain, "Second image:");
  // Calculate Frobenius norms of both input images before proceeding (for
  // normalization purposes and for the definition of error tolerance)
  const float norm_img1 =
          calc_frobenius_norm(img1, Q, n_rows, n_cols, col_stride_fwd_domain);
  const float norm_img2 =
          calc_frobenius_norm(img2, Q, n_rows, n_cols, col_stride_fwd_domain);
  // 1) Calculate the cross-correlation naively (for verification purposes);
  naive_cross_correlation(Q, n_rows, n_cols, col_stride_fwd_domain,
                          img1, img2, naive_corr).wait();
  // 2) Calculate the cross-correlation via Discrete Fourier Transforms (DFTs):
  //         corr = (1/num_elem) * iDFT(DFT(sig1) * CONJ(DFT(sig2)))
  // Initialize DFT descriptor
  oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                               oneapi::mkl::dft::domain::REAL>
    desc({n_rows, n_cols});
  // oneMKL DFT descriptors use unit scaling factors by default. Explicitly set
  // the non-default scaling factor for the backward ("inverse") DFT:
  desc.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                 1.0f / num_elem);
  desc.commit(Q);
  // --> desc operates in-place with the strides used herein by default.

  // Compute in-place forward transforms of both signals:
  // img1 <- DFT(img1)
  auto evt1 = oneapi::mkl::dft::compute_forward(desc, img1);
  // img2 <- DFT(img2)
  auto evt2 = oneapi::mkl::dft::compute_forward(desc, img2);
  // Compute the element-wise multipication of (complex) coefficients in
  // backward domain:
  // corr <- sig1 * CONJ(sig2) [component-wise]
  evt = oneapi::mkl::vm::mulbyconj(
          Q, n_rows * (n_cols / 2 + 1),
          reinterpret_cast<std::complex<float>*>(img1),
          reinterpret_cast<std::complex<float>*>(img2),
          reinterpret_cast<std::complex<float>*>(corr),
          {evt1, evt2});
  // Compute in-place (scaled) backward transform:
  // corr <- (1 / num_elem) * iDFT(corr)
  oneapi::mkl::dft::compute_backward(desc, corr, {evt}).wait();

  // Error bound for naive calculations:
  float max_err_threshold =
    2.0f * std::numeric_limits<float>::epsilon() * norm_img1 * norm_img2;
  // Adding an (empirical) error bound for the DFT-based calculation defined as
  //    epsilon * O(log(num_elem)) * scaling_factor * nrm2(input data),
  // wherein (for the last DFT at play)
  // - scaling_factor = 1.0 / num_elem;
  // - nrm2(input data) = norm_sig1[0] * norm_sig2[0] * num_elem
  // - O(log(num_elem)) ~ 2 * log(num_elem)
  //   [arbitrary choice; implementation-dependent behavior]
  max_err_threshold +=
    2.0f * std::log(static_cast<float>(num_elem))
         * std::numeric_limits<float>::epsilon() * norm_img1 * norm_img2;
  // Verify results by comparing DFT-based and naive calculations to each other,
  // and fetch optimal shift maximizing correlation (DFT-based calculation).
  float max_err = 0.0f;
  float max_corr = corr[0];
  std::pair<unsigned, unsigned> optimal_shift(0, 0);
  for (size_t s = 0; s < n_rows && return_code == 0; s++) {
    for (size_t p = 0; p < n_cols && return_code == 0; p++) {
      const float naive_val = naive_corr[s * col_stride_fwd_domain + p];
      const float dft_val   = corr[s * col_stride_fwd_domain + p];
      const float local_err = fabs(naive_val - dft_val);
      if (local_err > max_err)
        max_err = local_err;
      if (max_err > max_err_threshold) {
        std::cerr << "An error was found when verifying the results." << std::endl;
        std::cerr << "For shift value (s, p) = (" << s << ", " << p  << "):" << std::endl;
        std::cerr << "\tNaive calculation results in " << naive_val << std::endl;
        std::cerr << "\tFourier-based calculation results in " << dft_val << std::endl;
        std::cerr << "The error (" << max_err
                  << ") exceeds the threshold value of "
                  << max_err_threshold <<  std::endl;
        return_code = 1;
      }
      if (dft_val > max_corr) {
        max_corr = dft_val;
        optimal_shift.first   = s;
        optimal_shift.second  = p;
      }
    }
  }
  // Conclude:
  if (return_code == 0) {
    // Get average and standard deviation of either signal for normalizing the
    // correlation "score"
    const float avg_sig1 = img1[0] / num_elem;
    const float avg_sig2 = img2[0] / num_elem;
    const float std_dev_sig1 =
          std::sqrt((norm_img1 * norm_img1 - num_elem * avg_sig1 * avg_sig1) / num_elem);
    const float std_dev_sig2 =
          std::sqrt((norm_img2 * norm_img2 - num_elem * avg_sig2 * avg_sig2) / num_elem);
    const float normalized_corr =
      (max_corr / num_elem - avg_sig1 * avg_sig2) / (std_dev_sig1 * std_dev_sig2);
    std::cout << "Shift the second signal by translation vector ("
              << optimal_shift.first << ", " << optimal_shift.second
              << ") to get a maximum, normalized correlation score of "
              << normalized_corr
              << " (treating the signals as periodic along both dimensions)."
              << std::endl;
    std::cout << "Max difference between naive and Fourier-based calculations : "
              << max_err << " (verification threshold: " << max_err_threshold
              << ")." << std::endl;
  }
  // Cleanup
  sycl::free(img1, Q);
  sycl::free(img2, Q);
  sycl::free(corr, Q);
  sycl::free(naive_corr, Q);
  return return_code;
}
