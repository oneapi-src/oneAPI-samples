#pragma once

#include <sycl/sycl.hpp>
#include <complex>
#include "complex_helper.hpp"
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
namespace svecadd {
extern sycl::event svecadd(sycl::queue &, float, halide_buffer_t *, int, float, halide_buffer_t *, int, halide_buffer_t *);
}

namespace dvecadd {
extern sycl::event dvecadd(sycl::queue &, double, halide_buffer_t *, int, double, halide_buffer_t *, int, halide_buffer_t *);
}

namespace cvecadd {
extern sycl::event cvecadd(sycl::queue &, complexf, halide_buffer_t *, int, complexf, halide_buffer_t *, int, halide_buffer_t *);
}

namespace zvecadd {
extern sycl::event zvecadd(sycl::queue &, complexd, halide_buffer_t *, int, complexd, halide_buffer_t *, int, halide_buffer_t *);
}

// Query of the parameters of the systolic array (KKK) based on types
template <typename T>
constexpr auto get_systolic_array_dimensions() {
    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";
#ifdef TINY
    return 4;
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    if constexpr (std::is_same_v<T, float>) {
        return run_on_s10 ? 16 : 16;
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? 8 : 8;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? 8 : 8;
    } else {
        return run_on_s10 ? 4 : 4;
    }
#endif
}

} // namespace t2sp::blas::row_major
