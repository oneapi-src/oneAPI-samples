#pragma once

#include <cstdlib>
#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

// API of the reconfigurable matrix multiplication.The interface will be invoked by the GEMM implementation below.
#include "../reconfigurable_matmul/api.hpp"

// Data structures, etc. in Halide/T2SP
#include "Halide.h"
using namespace Halide;

namespace t2sp::blas::row_major {
// The API for GEMM. We choose the USM version of oneMKL DPC++/SYCL interface (https://oneapi-src.github.io/oneMKL/domains/blas/gemm.html) with the
// restriction of standard data types (s, d, c, z) only. In this case, the matrices, alpha and beta all have the same data type.
// So we define our GEMM interface as a template with a single type T.
template<typename T>
sycl::event gemm(sycl::queue &queue,
                 oneapi::mkl::transpose transa,
                 oneapi::mkl::transpose transb,
                 std::int64_t m,
                 std::int64_t n,
                 std::int64_t k,
                 T alpha,
                 const T *a,
                 std::int64_t lda,
                 const T *b,
                 std::int64_t ldb,
                 T beta,
                 T *c,
                 std::int64_t ldc,
                 const std::vector<sycl::event> &dependencies = {}) {
    bool TransposeA   = (transa == oneapi::mkl::transpose::T || transa == oneapi::mkl::transpose::C);
    bool TransposeB   = (transb == oneapi::mkl::transpose::T || transb == oneapi::mkl::transpose::C);
    bool ConjugateA   = (transa == oneapi::mkl::transpose::C);
    bool ConjugateB   = (transb == oneapi::mkl::transpose::C);
    bool SymmetricA   = false;
    bool HermitianA   = false;
    bool UpA          = false;
    bool SymmetricB   = false;
    bool HermitianB   = false;
    bool UpB          = false;
    bool SymmetricC   = false;
    bool HermitianC   = false;
    bool UpC          = false;
    bool HalfSpaceOut = false;

    // Check parameters for constraints set by oneMKL DPC++ interface
    _halide_user_assert(m >= 0 && n >= 0 && k >= 0) << "m = " << m << ", n = " << n << ", k = " << k;
    _halide_user_assert(a && b && c) << "a = " << (const void *)a << ", b = " << (const void *)b << ", c = " << (const void *)c;
    _halide_user_assert(!TransposeA && lda >= k || TransposeA && lda >= m)
        << std::boolalpha << "TransposeA = " << TransposeA << ", lda = " << lda << ", m = " << m << ", k = " << k;
    _halide_user_assert(!TransposeB && ldb >= n || TransposeB && ldb >= k)
        << std::boolalpha << "TransposeB = " << TransposeB << ", ldb = " << ldb << ", n = " << n << ", k = " << k;
    _halide_user_assert(ldc >= n) << "ldc = " << ldc << ", n = " << n;

    _halide_user_assert((std::is_same_v<float, T>) ||
                        (std::is_same_v<double, T>) ||
                        (std::is_same_v<std::complex<float>, T>) ||
                        (std::is_same_v<std::complex<double>, T>)) << "Unsupported data type";

    const auto [KKK, JJJ, III, JJ, II, KK] = get_systolic_array_dimensions<T>();

    // TOREMOVE: These two constraints below should be checked by the reconfigurable matmul instead.
    _halide_user_assert(n % JJJ == 0) << "The current implementation requires that n must be a multiple of " << JJJ
                              << "(the output matrix is stored in " << JJJ << "-wide vectors for memory efficiency), but n = " << n;
    _halide_user_assert(k % KKK == 0) << "The current implementation requires that k must be a multiple of " << KKK
                              << "(the input matrices are loaded in " << KKK << "-wide vectors for memory efficiency), but k = " << k;

    using Halide::Runtime::Buffer;
    halide_dimension_t dim_a[]{{0, TransposeA ? m : k, 1}, {0, TransposeA ? k : m, lda}};
    Buffer<T> A_buffer{const_cast<T *>(a), 2, dim_a};
    halide_dimension_t dim_b[]{{0, TransposeB ? k : n, 1}, {0, TransposeB ? n : k, ldb}};
    Buffer<T> B_buffer{const_cast<T *>(b), 2, dim_b};
    halide_dimension_t dim_c[]{{0, n, 1}, {0, m, ldc}};
    Buffer<T> C_buffer{c, 2, dim_c};
    Buffer<T> Output_buffer{JJJ, JJ, II, III, (n + (JJJ * JJ - 1)) / (JJJ * JJ), (m + (III * II - 1)) / (III * II)};

    for (sycl::event e : dependencies) {
        e.wait();
    }

    sycl::event done;
    if constexpr (std::is_same_v<float, T>) {
        done = t2sp::blas::row_major::ssssmatmul::ssssmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
    } else if constexpr (std::is_same_v<double, T>) {
        done = t2sp::blas::row_major::ddddmatmul::ddddmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
    } else if constexpr (std::is_same_v<std::complex<float>, T>) {
        done = t2sp::blas::row_major::ccccmatmul::ccccmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
    } else {
        done = t2sp::blas::row_major::zzzzmatmul::zzzzmatmul(queue, A_buffer, TransposeA, ConjugateA, SymmetricA, HermitianA, UpA,
                                                                    B_buffer, TransposeB, ConjugateB, SymmetricB, HermitianB, UpB,
                                                                    C_buffer,                         SymmetricC, HermitianC, UpC,
                                                                    HalfSpaceOut, alpha, beta, Output_buffer);
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
