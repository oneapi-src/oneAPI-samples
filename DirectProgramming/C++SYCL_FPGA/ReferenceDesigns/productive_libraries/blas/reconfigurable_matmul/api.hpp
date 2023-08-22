#pragma once

#include <sycl/sycl.hpp>
#include "Halide.h"
#include "complex_helper.hpp"
using namespace Halide;

namespace t2sp::blas::row_major {
// All the legal combinations in the order of TA, TB, TC and TS.
namespace ssssmatmul {
extern sycl::event ssssmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, float alpha, float beta, struct halide_buffer_t *Output_buffer);
}

namespace ddddmatmul {
extern sycl::event ddddmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, double alpha, double beta, struct halide_buffer_t *Output_buffer);
}

namespace ccccmatmul {
extern sycl::event ccccmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, complexf alpha, complexf beta, struct halide_buffer_t *Output_buffer);
}

namespace zzzzmatmul {
extern sycl::event zzzzmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, complexd alpha, complexd beta, struct halide_buffer_t *Output_buffer);
}

// For HERK, the scalar type is real or double, different from the matrices' data types.
namespace cccsmatmul {
extern sycl::event cccsmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, float alpha, float beta, struct halide_buffer_t *Output_buffer);
}

namespace zzzdmatmul {
extern sycl::event zzzdmatmul(sycl::queue &q_device, struct halide_buffer_t *A_buffer, bool TransposeA, bool ConjugateA, bool SymmetricA, bool HermitianA, bool UpA,
                                                     struct halide_buffer_t *B_buffer, bool TransposeB, bool ConjugateB, bool SymmetricB, bool HermitianB, bool UpB,
                                                     struct halide_buffer_t *C_buffer,                                   bool SymmetricC, bool HermitianC, bool UpC,
                                                     bool HalfSpaceOut, double alpha, double beta, struct halide_buffer_t *Output_buffer);
}


// Query of the parameters of the systolic array (KKK, JJJ, III, JJ, II, KK) based on types
template <typename T>
constexpr auto get_systolic_array_dimensions() {
    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";
#ifdef TINY
    return std::tuple{4, 4, 4, 4, 4, 4};
#else
#ifdef S10
    constexpr bool run_on_s10 = true;
#else
    constexpr bool run_on_s10 = false;
#endif
    if constexpr (std::is_same_v<T, float>) {
        return run_on_s10 ? std::tuple{16, 16, 10, 32, 32, 32} : std::tuple{16, 8, 10, 32, 32, 32};
    } else if constexpr (std::is_same_v<T, double>) {
        return run_on_s10 ? std::tuple{8, 4, 6, 32, 32, 32} : std::tuple{8, 4, 6, 32, 32, 32};
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return run_on_s10 ? std::tuple{8, 4, 10, 32, 32, 32} : std::tuple{8, 4, 10, 32, 32, 32};
    } else {
        return run_on_s10 ? std::tuple{4, 6, 3, 32, 32, 32} : std::tuple{4, 4, 3, 32, 32, 32};
    }
#endif
}

} // namespace t2sp::blas::row_major
