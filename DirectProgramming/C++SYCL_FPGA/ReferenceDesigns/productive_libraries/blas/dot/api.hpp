#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable dot. The interface will be invoked by the DOT implementation below.
#include "../reconfigurable_dotprod/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for DOT. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/dot.html).
template<typename T_res, typename T>
sycl::event dot(sycl::queue &queue,
                 std::int64_t n,
                 const T *x,
                 std::int64_t incx,
                 const T *y,
                 std::int64_t incy,
                 T_res *result,
                 const std::vector<sycl::event> &dependencies = {}) {

    _halide_user_assert((std::is_same_v<float, T_res> && std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T_res> &&
                        (std::is_same_v<double, T> || std::is_same_v<float, T>))) << "Unsupported data type";

    const auto KKK = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % KKK == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << KKK
                                      << "(the vectorized dimension for the input vectors), but n = " << n;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_x[]{{0, n, std::abs(incx)}, {0, 1, 1}};
    halide_dimension_t dim_y[]{{0, n, std::abs(incy)}, {0, 1, 1}};
    halide_dimension_t dim_res[]{{0, 1, 1}};

    Buffer<T> X_buffer{const_cast<T *>(x), 2, dim_x};
    Buffer<T> Y_buffer{const_cast<T *>(y), 2, dim_y};
    Buffer<T_res> Res_buffer{result, 1, dim_res};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool ConjugateX = false;
    bool SignBitY = false;
    bool SqrtRet = false;

    sycl::event done;

    if constexpr (std::is_same_v<float, T_res>) {
        done = t2sp::blas::row_major::sdotprod::sdotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    } else if constexpr (std::is_same_v<double, T_res> && std::is_same_v<double, T>){
        done = t2sp::blas::row_major::ddotprod::ddotprod(queue, ConjugateX,
                                                 X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                 Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    } else {
        done = t2sp::blas::row_major::dsdotprod::dsdotprod(queue, ConjugateX,
                                                  X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                  Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    }
    return done;
}

} // namespace t2sp::blas::row_major
