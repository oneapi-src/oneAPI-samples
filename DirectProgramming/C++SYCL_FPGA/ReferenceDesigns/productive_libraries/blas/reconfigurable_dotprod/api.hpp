#pragma once

#include <sycl/sycl.hpp>
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
namespace sdotprod {
extern sycl::event sdotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

namespace ddotprod {
extern sycl::event ddotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

namespace cdotprod {
extern sycl::event cdotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

namespace zdotprod {
extern sycl::event zdotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

namespace sdsdotprod {
extern sycl::event sdsdotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

namespace dsdotprod {
extern sycl::event dsdotprod(sycl::queue &, bool, halide_buffer_t *, int, bool, halide_buffer_t *, int, bool, halide_buffer_t *);
}

// Query of the KKK parameter of the systolic array based on the input vectors' type
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
        return run_on_s10 ? 32 : 16;
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? 16 : 8;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? 16 : 8;
    } else {
        return run_on_s10 ? 8 : 4;
    }
#endif
}

} // namespace t2sp::blas::row_major
