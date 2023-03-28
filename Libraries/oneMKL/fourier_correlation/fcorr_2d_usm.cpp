//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
//  Content:
//     This code implements the 2D Fourier correlation algorithm
//     using SYCL, oneMKL, oneDPL, and unified shared memory
//     (USM).
//
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>
#include <oneapi/mkl/dfti.hpp>
#include <oneapi/mkl/vm.hpp>
#include <mkl.h>

#include <iostream>

int main(int argc, char **argv)
{
    // Initialize SYCL queue
    sycl::queue Q(sycl::default_selector_v);
    std::cout << "Running on: "
              << Q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Allocate 2D image and correlation arrays
    unsigned int n_rows = 8, n_cols = 8;
    auto img1 = sycl::malloc_shared<float>(n_rows * n_cols * 2 + 2, Q);
    auto img2 = sycl::malloc_shared<float>(n_rows * n_cols * 2 + 2, Q);
    auto corr = sycl::malloc_shared<float>(n_rows * n_cols * 2 + 2, Q);

    // Initialize input images with artificial data. Do initialization on the device.
    // Generalized strides for row-major addressing
    int r_stride = 1, c_stride = (n_cols / 2 + 1) * 2, c_stride_h = (n_cols / 2 + 1);
    Q.parallel_for<>(sycl::range<2>{n_rows, n_cols}, [=](sycl::id<2> idx) {
        unsigned int r = idx[0];
        unsigned int c = idx[1];
        img1[r * c_stride + c * r_stride] = 0.0;
        img2[r * c_stride + c * r_stride] = 0.0;
        corr[r * c_stride + c * r_stride] = 0.0;
    }).wait();
    Q.single_task<>([=]() {
        // Set a box of elements in the lower right of the first image
        img1[4 * c_stride + 5 * r_stride] = 1.0;
        img1[4 * c_stride + 6 * r_stride] = 1.0;
        img1[5 * c_stride + 5 * r_stride] = 1.0;
        img1[5 * c_stride + 6 * r_stride] = 1.0;

        // Set a box of elements in the upper left of the second image
        img2[1 * c_stride + 1 * r_stride] = 1.0;
        img2[1 * c_stride + 2 * r_stride] = 1.0;
        img2[2 * c_stride + 1 * r_stride] = 1.0;
        img2[2 * c_stride + 2 * r_stride] = 1.0;
    }).wait();

    std::cout << std::endl << "First image:" << std::endl;
    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            std::cout << img1[r * c_stride + c * r_stride] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl << "Second image:" << std::endl;
    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            std::cout << img2[r * c_stride + c * r_stride] << " ";
        }
        std::cout << std::endl;
    }

    // Initialize FFT descriptor
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::REAL>
        forward_plan({n_rows, n_cols});

    // Data layout in real domain
    std::int64_t real_layout[4] = {0, c_stride, 1};
    forward_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                           real_layout);

    // Data layout in conjugate-even domain
    std::int64_t complex_layout[4] = {0, c_stride_h, 1};
    forward_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                           complex_layout);

    // Perform forward transforms on real arrays
    forward_plan.commit(Q);
    auto evt1 = oneapi::mkl::dft::compute_forward(forward_plan, img1);
    auto evt2 = oneapi::mkl::dft::compute_forward(forward_plan, img2);

    // Compute: DFT(img1) * CONJG(DFT(img2))
    oneapi::mkl::vm::mulbyconj(Q, n_rows * c_stride_h,
                               reinterpret_cast<std::complex<float>*>(img1),
                               reinterpret_cast<std::complex<float>*>(img2),
                               reinterpret_cast<std::complex<float>*>(corr),
                               {evt1, evt2})
                               .wait();

    // Perform backward transform on complex correlation array
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::REAL>
        backward_plan({n_rows, n_cols});

    // Data layout in conjugate-even domain
    backward_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                            complex_layout);

    // Data layout in real domain
    backward_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                            real_layout);

    backward_plan.commit(Q);
    auto bwd = oneapi::mkl::dft::compute_backward(backward_plan, corr);
    bwd.wait();

    std::cout << std::endl << "Normalized Fourier correlation result:"
              << std::endl;
    for (unsigned int r = 0; r < n_rows; r++) {
        for (unsigned int c = 0; c < n_cols; c++) {
            std::cout << corr[r * c_stride + c * r_stride] / (n_rows * n_cols)
                      << " ";
        }
        std::cout << std::endl;
    }

    // Find the translation vector that gives maximum correlation value
    auto policy = oneapi::dpl::execution::make_device_policy(Q);
    auto maxloc = oneapi::dpl::max_element(policy,
                                           corr,
                                           corr + (n_rows * n_cols * 2 + 2));
    policy.queue().wait();
    auto s = oneapi::dpl::distance(corr, maxloc);
    float max_corr = corr[s];
    int x_shift = s % (n_cols + 2);
    int y_shift = s / (n_rows + 2);

    std::cout << std::endl << "Shift the second image (x, y) = (" << x_shift
         << ", " << y_shift
         << ") elements relative to the first image to get a maximum,"
         << std::endl << "normalized correlation score of "
         << max_corr / (n_rows * n_cols)
         << ". Treat the images as circularly shifted versions of each other."
         << std::endl;

    // Cleanup
    sycl::free(img1, Q);
    sycl::free(img2, Q);
    sycl::free(corr, Q);
}
