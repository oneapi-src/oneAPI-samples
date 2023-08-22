#include <cstdlib>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

// The HEMM API to invoke
#include "./api.hpp"

// Useful routines from the OneMKL unit tests
#include "allocator_helper.hpp"
#include "oneapi/mkl.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include "exception_handler.hpp"
#include "matrix_multiply_statistics.hpp"

using namespace std;

template <typename T>
void test(oneapi::mkl::side left_right, oneapi::mkl::uplo upper_lower, int m, int n, int lda, int ldb, int ldc, T alpha, T beta) {
    vector<T, allocator_helper<T, 64>> a, b;
    vector<T, allocator_helper<T, 64>> c, c_ref;
    rand_matrix(a, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, left_right == oneapi::mkl::side::left ? m : n,
                                                                              left_right == oneapi::mkl::side::left ? m : n, lda);
    // set the elements on the diagonal to real numbers
    for (int i = 0; i < (left_right == oneapi::mkl::side::left ? m : n); i++) {
        a[i + i * lda] = a[i + i * lda].real();
    }

    rand_matrix(b, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, m, n, ldb);
    rand_matrix(c, oneapi::mkl::layout::row_major, oneapi::mkl::transpose::N, m, n, ldc);
    c_ref = c;

    // Create a queue on an FPGA device.
    sycl::queue q_device(sycl::ext::intel::fpga_selector_v, fpga_tools::exception_handler, sycl::property::queue::enable_profiling());

    sycl::event e = t2sp::blas::row_major::hemm(q_device, left_right, upper_lower, m, n, alpha, a.data(), lda,
                                                b.data(), ldb, beta, c.data(), ldc);
    e.wait();

    // Statistics for performance measurement
    uint64_t start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    uint64_t exec_time = end - start;
    int k = (left_right == oneapi::mkl::side::left ? m : n);
    double   total_flops, total_bytes;
    matrix_multiply_statistics<T>(m, n, k, exec_time, total_flops, total_bytes);
    std::cout << "FP operations: " << total_flops << "\n";
    std::cout << "Execution time: " << exec_time << " ns\n";
    std::cout << "GFLOPs: " << total_flops / exec_time << "\n";
    std::cout << "Memory bytes: " << total_bytes << "\n";
    std::cout << "Size of matrix a: " << m << " * " << k << "\n";
    std::cout << "Size of matrix b: " << k << " * " << n << "\n";
    std::cout << "Size of matrix c: " << m << " * " << n << "\n";
}

int main() {
#if defined(PREFIX_C)
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<float>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    std::complex<float> alpha = {2.0f, -0.5f};
    std::complex<float> beta  = {3.0f, -1.5f};
    test<std::complex<float>>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#else
    const auto [KKK, JJJ, III, JJ, II, KK] = t2sp::blas::row_major::get_systolic_array_dimensions<std::complex<double>>();
    int64_t m = III * II * 32;
    int64_t n = JJJ * JJ * 32;
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = n;
    std::complex<double> alpha = {2.0f, -0.5f};
    std::complex<double> beta  = {3.0f, -1.5f};
    test<std::complex<double>>(oneapi::mkl::side::L, oneapi::mkl::uplo::U, m, n, lda, ldb, ldc, alpha, beta);
#endif
}
