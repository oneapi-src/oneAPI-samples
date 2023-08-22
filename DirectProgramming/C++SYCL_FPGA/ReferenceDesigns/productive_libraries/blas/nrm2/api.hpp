#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable dot product. The interface will be invoked by the NRM2 implementation below.
#include "../reconfigurable_dotprod/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for NRM2. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/nrm2.html)
template<typename T, typename T_res>
sycl::event nrm2(sycl::queue &queue,
                 std::int64_t n,
                 const T *x,
                 std::int64_t incx,
                 T_res *result,
                 const std::vector<sycl::event> &dependencies = {}) {

    _halide_user_assert((std::is_same_v<float, T> ||
                        std::is_same_v<double, T> ||
                        std::is_same_v<std::complex<float>, T> ||
                        std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto KKK = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % KKK == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << KKK
                              << "(the vectorized dimension for the input vectors), but n = " << n;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_x[]{{0, n, std::abs(incx)}, {0, 1, 1}};
    halide_dimension_t dim_res[]{{0, 1, 1}};

    Buffer<T> X_buffer{const_cast<T *>(x), 2, dim_x};

    std::unique_ptr<T> result_buffer{new T{}};
    Buffer<T> Res_buffer{result_buffer.get(), 1, dim_res};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool ConjugateX = true;
    bool SignBitY = false;
    bool SqrtRet = true;

    sycl::event done;

    if constexpr (std::is_same_v<float, T>) {
        done = t2sp::blas::row_major::sdotprod::sdotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SqrtRet, Res_buffer);
    } else if constexpr (std::is_same_v<double, T>) {
        done = t2sp::blas::row_major::ddotprod::ddotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SqrtRet, Res_buffer);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::cdotprod::cdotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SqrtRet, Res_buffer);
    } else {
        done = t2sp::blas::row_major::zdotprod::zdotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SqrtRet, Res_buffer);
    }
    done.wait();
    if constexpr (std::is_same_v<std::complex<float>, T> ||
                  std::is_same_v<std::complex<double>, T>) {
        *result = result_buffer->real();
    } else {
        *result = *result_buffer;
    }
    return done;
}

} // namespace t2sp::blas::row_major
