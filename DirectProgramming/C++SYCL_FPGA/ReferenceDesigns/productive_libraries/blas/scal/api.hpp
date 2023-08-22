#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable vecadd. The interface will be invoked by the SCAL implementation below.
#include "../reconfigurable_vecadd/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for SCAL. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/scal.html) with the
// restriction of standard data types (s, d, c, z) only.
template<typename T>
sycl::event scal(sycl::queue &queue,
                 std::int64_t n,
                 T alpha,
                 T *x,
                 std::int64_t incx,
                 const std::vector<sycl::event> &dependencies = {}) {

    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto KKK = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % KKK == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << KKK
                              << "(the vectorized dimension for the input vectors), but n = " << n;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_x[]{{0, n, std::abs(incx)}, {0, 1, 1}};

    Buffer<T> X_buffer{x, 2, dim_x};
    Buffer<T> Res_buffer(KKK, n / KKK, 1);

    for (sycl::event e : dependencies) {
        e.wait();
    }

    auto Alpha = alpha;
    auto Beta = 0.0;

    sycl::event done;

    if constexpr (std::is_same_v<float, T>) {
        done = t2sp::blas::row_major::svecadd::svecadd(queue, Alpha,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Beta,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Res_buffer);
    } else if constexpr (std::is_same_v<double, T>) {
        done = t2sp::blas::row_major::dvecadd::dvecadd(queue, Alpha,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Beta,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Res_buffer);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::cvecadd::cvecadd(queue, Alpha,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Beta,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Res_buffer);
    } else {
        done = t2sp::blas::row_major::zvecadd::zvecadd(queue, Alpha,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Beta,
                                                       X_buffer, std::abs(static_cast<int>(incx)), Res_buffer);
    }
    done.wait();
    for (auto k = 0; k < n / KKK; k++) {
        for (auto kkk = 0; kkk < KKK; kkk++)
        x[(kkk + k * KKK) * std::abs(incx)] = Res_buffer(kkk, k, 0);
    }
    return done;
}

} // namespace t2sp::blas::row_major
