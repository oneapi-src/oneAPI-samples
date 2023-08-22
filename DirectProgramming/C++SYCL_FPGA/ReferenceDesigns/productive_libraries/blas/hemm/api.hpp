#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the HEMM implementation below.
#include "../reconfigurable_matmul/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for HEMM. We choose the USM version of oneMKL DPC++ interface (https://oneapi-src.github.io/oneMKL/domains/blas/hemm.html).
template<typename T>
sycl::event hemm(sycl::queue &queue,
                 oneapi::mkl::side left_right,
                 oneapi::mkl::uplo upper_lower,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 const T* a,
                 std::int64_t lda,
                 const T* b,
                 std::int64_t ldb,
                 T beta,
                 T* c,
                 std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {}) {
    // Check parameters for constraints set by oneMKL DPC++ interface
    _halide_user_assert(m >= 0 && n >= 0) << "m = " << m << ", n = " << n;
    _halide_user_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    _halide_user_assert(lda > 0 && (left_right == oneapi::mkl::side::L ? lda >= m : lda >= n)) << "lda = " << lda << ", m = " << m << ", n = " << n;
    _halide_user_assert(ldb > 0 && ldb >= n) << "ldb = " << ldb << ", n = " << n;
    _halide_user_assert(ldc > 0 && ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert((std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto [KKK, JJJ, III, JJ, II, KK] = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % JJJ == 0) << "The current implementation requires that n must be a multiple of " << JJJ
                              << "(the output matrix is stored in " << JJJ << "-wide vectors for memory efficiency), but n = " << n;
    _halide_user_assert((left_right == oneapi::mkl::side::L ? m : n) % KKK == 0) << "The current implementation requires that reduction dimension must be a multiple of " << KKK
                              << "(the input matrices are loaded in " << KKK << "-wide vectors for memory efficiency), but the reduction dimension = " << (left_right == oneapi::mkl::side::L ? m : n);

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, left_right == oneapi::mkl::side::L ? m : n, 1}, {0, left_right == oneapi::mkl::side::L ? m : n, lda}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, n, 1}, {0, m, ldb}};
    Buffer<T> B_buffer{const_cast<T *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, m, ldc}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (m + (III * II - 1)) / (III * II)};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    bool TransposeA   = false;
    bool ConjugateA   = false;
    bool SymmetricA   = false;
    bool HermitianA   = true;
    bool UpA          = (upper_lower == oneapi::mkl::uplo::U);
    bool TransposeB   = false;
    bool ConjugateB   = false;
    bool SymmetricB   = false;
    bool HermitianB   = false;
    bool UpB          = false;
    bool SymmetricC   = false;
    bool HermitianC   = false;
    bool UpC          = false;
    bool HalfSpaceOut = false;

    sycl::event done;
    if constexpr (std::is_same_v<std::complex<float>, T>) {
        if (left_right == oneapi::mkl::side::L) {
            done = t2sp::blas::row_major::ccccmatmul::ccccmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                        B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                        C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                        HalfSpaceOut, alpha, beta, Output_buffer);
        } else {
            done = t2sp::blas::row_major::ccccmatmul::ccccmatmul(queue, B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                        A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                        C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                        HalfSpaceOut, alpha, beta, Output_buffer);
        }
    } else {
        if (left_right == oneapi::mkl::side::L) {
            done = t2sp::blas::row_major::zzzzmatmul::zzzzmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                        B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                        C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                        HalfSpaceOut, alpha, beta, Output_buffer);
        } else {
            done = t2sp::blas::row_major::zzzzmatmul::zzzzmatmul(queue, B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                        A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                        C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                        HalfSpaceOut, alpha, beta, Output_buffer);
        }
    }
    for (int i = 0; i < (m + (III * II - 1)) / (III * II); i++) {
        for (int j = 0; j < (n + (JJJ * JJ - 1)) / (JJJ * JJ); j++) {
            for (int ii = 0; ii < II; ii++) {
                for (int jj = 0; jj < JJ; jj++) {
                    for (int iii = 0; iii < III; iii++) {
                        for (int jjj = 0; jjj < JJJ; jjj++) {
                            int total_i = iii + III * ii + III * II * i;
                            int total_j = jjj + JJJ * jj + JJJ * JJ * j;
                            if (total_i < m && total_j < n) {
                                c[total_j + total_i * ldc] = Output_buffer(jjj, jj, ii, iii, j, i);
                            }
                        }
                    }
                }
            }
        }
    }
    return done;
}

} // namespace t2sp::blas::row_major
