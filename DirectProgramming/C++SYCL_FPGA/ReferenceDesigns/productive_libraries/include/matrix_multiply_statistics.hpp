#pragma once
#include <complex>
#include <sycl/sycl.hpp>

template <typename T>
void matrix_multiply_statistics(int64_t m, int64_t n, int64_t k, uint64_t exec_time, double &total_flops, double &total_bytes) {
    double multiplications_in_product, additions_in_product, multiplications_in_sum, additions_in_sum, flops_per_multiplication, flops_per_addition, total_data, bytes_per_data;
    // m*n results in computing product=op(A)*op(B), each result is reduced from k number of multiplications and additions.
    multiplications_in_product = m * n * k;
    additions_in_product = m * n * k;
    // m*n results in computing alpha*product + beta*C, each is reduced from two multiplications and one addition.
    multiplications_in_sum = 2.0 * m * n;
    additions_in_sum = m * n;
    if ((std::is_same_v<float, T> || std::is_same_v<double, T>)) {
        // For float and double, 1 multiplication/addition is 1 FP operation (of single- or double-precision)
        flops_per_multiplication = 1;
        flops_per_addition = 1;
    } else {
        // For complex float and double, 1 multiplication of two complex numbers requires 4 FP MUL and 2 FP ADD operations (of single- or double-precision)
        // 1 addition of two complex numbers requires 2 FP ADD operations (of single- or double-precision)
        flops_per_multiplication = 6;
        flops_per_addition = 2;
    }
    total_flops = multiplications_in_product * flops_per_multiplication + additions_in_product * flops_per_addition +
                  multiplications_in_sum     * flops_per_multiplication + additions_in_sum     * flops_per_addition;

    total_data = m * k + k * n + 2.0 * m * n; // data accessed from op(A), op(B), original C, and final C
    bytes_per_data = sizeof(T);
    total_bytes = total_data * bytes_per_data;
}
