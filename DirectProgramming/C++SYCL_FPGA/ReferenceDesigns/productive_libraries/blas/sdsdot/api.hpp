#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable dot. The interface will be invoked by the SDSDOT implementation below.
#include "../reconfigurable_dotprod/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for SDSDOT. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/sdsdot.html)
sycl::event sdsdot(sycl::queue &queue,
                   std::int64_t n,
                   float sb,
                   const float *x,
                   std::int64_t incx,
                   const float *y,
                   std::int64_t incy,
                   float *result,
                   const std::vector<sycl::event> &dependencies = {}) {
    // TODO: handle the special case that n < 0

    const auto KKK = get_systolic_array_dimensions<float>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % KKK == 0) << "For performance reasons, the current implementation requires that n must be a multiple of " << KKK
                              << "(the vectorized dimension for the input vectors), but n = " << n;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_x[]{{0, n, std::abs(incx)}, {0, 1, 1}};
    halide_dimension_t dim_y[]{{0, n, std::abs(incy)}, {0, 1, 1}};
    halide_dimension_t dim_res[]{{0, 1, 1}};

    Buffer<float> X_buffer{const_cast<float *>(x), 2, dim_x};
    Buffer<float> Y_buffer{const_cast<float *>(y), 2, dim_y};

    Buffer<double> Res_buffer(1);

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool ConjugateX = false;
    bool SignBitY = false;
    bool SqrtRet = false;

    sycl::event done;

    done = t2sp::blas::row_major::sdsdotprod::sdsdotprod(queue, ConjugateX,
                                                   X_buffer, std::abs(static_cast<int>(incx)), SignBitY,
                                                   Y_buffer, std::abs(static_cast<int>(incy)), SqrtRet, Res_buffer);
    done.wait();
    Res_buffer(0) += sb;
    *result = static_cast<float>(Res_buffer(0));
    return done;
}

} // namespace t2sp::blas::row_major
