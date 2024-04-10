#ifndef __SVD_TESTCASE__
#define __SVD_TESTCASE__

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "svd.hpp"
#include "svd_testbench_tool.hpp"
#include "print_matrix.hpp"

template <typename T, unsigned rows_A, unsigned cols_A, bool is_complex = false>
struct SVDTestcase {
  std::vector<std::vector<T>> input_A;
  std::vector<T> output_S;
  T S_error;
  T A_error;
  T U_orthogonal_error;
  T V_orthogonal_error;
  double delta_time;
  double throughput;

  SVDTestcase(std::vector<std::vector<T>> A, std::vector<T> S)
      : input_A(A), output_S(S) {}

  std::vector<T> col_major_A() {
    std::vector<T> flat_A;

    for (int col = 0; col < cols_A; col++) {
      for (int row = 0; row < rows_A; row++) {
        flat_A.push_back(input_A[row][col]);
      }
    }
    return flat_A;
  }

  std::vector<T> ExtractSingularValue(std::vector<T> mat_S) {
    std::vector<T> singular_value;
    constexpr int kSingularValueSize = rows_A < cols_A ? rows_A : cols_A;
    for (int i = 0; i < kSingularValueSize; i++) {
      singular_value.push_back(mat_S[i * rows_A + i]);
    }
    return singular_value;
  }

  T CompareS(std::vector<T> input_vec) {
    T max_diff = 0.0;
    // in case singular values are not sorted
    std::sort(std::begin(input_vec), std::end(input_vec), std::greater<>());
    for (int i = 0; i < output_S.size(); i++) {
      T cur_diff = abs(input_vec[i] - output_S[i]);
      if (cur_diff > max_diff) max_diff = cur_diff;
    }
    S_error = max_diff;
    return max_diff;
  }

  T CheckUSV(std::vector<T> flat_A, std::vector<T> flat_U,
              std::vector<T> flat_S, std::vector<T> flat_V) {
    // U @ S
    std::vector<T> US(rows_A * cols_A, 0);
    svd_testbench_tool::soft_matmult<T>(flat_U, rows_A, rows_A, flat_S, rows_A,
                                        cols_A, US);
    // transpose to get Vt
    std::vector<T> Vt(cols_A * cols_A, 0);
    svd_testbench_tool::soft_transpose<T>(flat_V, cols_A, cols_A, Vt);
    // US @ Vt
    std::vector<T> USV(rows_A * cols_A, 0);
    svd_testbench_tool::soft_matmult<T>(US, rows_A, cols_A, Vt, cols_A, cols_A,
                                        USV);
    // svd_testbench_tool::print_matrix<T>(USV, rows_A, cols_A);
    T max_diff = 0.0;
    for (int i = 0; i < (rows_A * cols_A); i++) {
      T cur_diff = abs(USV[i] - flat_A[i]);
      if (cur_diff > max_diff) max_diff = cur_diff;
    }
    A_error = max_diff;
    return max_diff;
  }

  T CheckOrthogonal(std::vector<T> flat_mat, unsigned rows, unsigned cols) {
    // check mat @ mat transpose == identity
    std::vector<T> mat_t(cols * rows, 0);
    std::vector<T> mat_i(rows * rows, 0);
    svd_testbench_tool::soft_transpose<T>(flat_mat, rows, cols, mat_t);
    svd_testbench_tool::soft_matmult<T>(flat_mat, rows, cols, mat_t, cols, rows,
                                        mat_i);
    T max_diff = 0.0;
    for (int i = 0; i < (rows * rows); i++) {
      int cur_col = int(i / rows);
      int cur_row = i % rows;
      T cur_diff = 0.0;
      if (cur_row == cur_col) {
        cur_diff = abs(mat_i[i] - 1.0);
      } else {
        cur_diff = abs(mat_i[i]);
      }

      if (cur_diff > max_diff) max_diff = cur_diff;
    }
    return max_diff;
  }

  template <unsigned k_fixed_iteration = FIXED_ITERATIONS,
            unsigned k_raw_latency = 110, int k_zero_threshold_1e = -8>
  T RunTest(sycl::queue q, int benchmark_rep = 1, bool print_matrices = false) {
    std::vector<T> flat_A = col_major_A();
    std::vector<T> flat_U(rows_A * rows_A);
    std::vector<T> flat_S(rows_A * cols_A);
    std::vector<T> flat_V(cols_A * cols_A);
    std::vector<ac_int<1, false>> rank_deficient(1);

    std::cout << "Running SVD test with input size " << rows_A << " x "
              << cols_A << ", repeating " << benchmark_rep << " time(s)"
              << std::endl;

    delta_time =
        SingularValueDecomposition<rows_A, cols_A, k_fixed_iteration,
                                   k_raw_latency, k_zero_threshold_1e, T>(
            flat_A, flat_U, flat_S, flat_V, rank_deficient, q, 1, benchmark_rep);

    throughput = (benchmark_rep / delta_time);

    CompareS(ExtractSingularValue(flat_S));

    A_error = CheckUSV(flat_A, flat_U, flat_S, flat_V);
    U_orthogonal_error = CheckOrthogonal(flat_U, rows_A, rows_A);
    V_orthogonal_error = CheckOrthogonal(flat_V, cols_A, cols_A);

    if (print_matrices) {
      std::cout << "S:\n";
      // print_matrix(ExtractSingularValue(flat_S), 1, rows_A);
      svd_testbench_tool::print_matrix<T>(flat_S, rows_A, cols_A, true);
      std::cout << "V:\n";
      svd_testbench_tool::print_matrix<T>(flat_V, cols_A, cols_A, true);
      std::cout << "U:\n";
      svd_testbench_tool::print_matrix<T>(flat_U, rows_A, rows_A, true);
      std::cout << "Rank deficient input: " 
        << (rank_deficient[0] ? "True" : "False") << std::endl;
    }
    return std::max({S_error, A_error, U_orthogonal_error, V_orthogonal_error});
  }

  void PrintResult() {
    std::cout << "Singular value differences: " << S_error << std::endl;
    std::cout << "Decomposition differences (A = USVt): " << A_error << std::endl;
    std::cout << "U orthogonal differences: " << U_orthogonal_error << std::endl;
    std::cout << "V orthogonal differences: " << V_orthogonal_error << std::endl;
    std::cout << "Total duration: " << delta_time << "s" << std::endl;
    std::cout << "Throughput: " << throughput * 1e-3 << "k matrices/s"
              << std::endl;
  }
};


#endif  // __SVD_TESTCASE__