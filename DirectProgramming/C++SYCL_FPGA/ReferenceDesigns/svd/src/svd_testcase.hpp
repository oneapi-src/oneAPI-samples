#ifndef __SVD_TESTCASE__
#define __SVD_TESTCASE__

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "svd.hpp"
#include "svd_helper.hpp"

template <typename T, unsigned rows_A, unsigned cols_A, bool is_complex = false>
struct SVDTestcase {
  std::vector<std::vector<T>> input_A;
  std::vector<T> output_S;
  T S_error;
  float S_error_r;
  T A_error;
  float A_error_r;
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

  std::vector<T> extract_singular_value(std::vector<T> mat_S) {
    std::vector<T> singular_value;
    int singular_value_size = rows_A < cols_A ? rows_A : cols_A;
    for (int i = 0; i < singular_value_size; i++) {
      singular_value.push_back(mat_S[i * rows_A + i]);
    }
    return singular_value;
  }

  T compare_S(std::vector<T> input_vec) {
    T max_diff = 0.0;
    float max_ratio = 0.0;
    // in case singuler values are not sorted
    std::sort(std::begin(input_vec), std::end(input_vec), std::greater<>());
    for (int i = 0; i < output_S.size(); i++) {
      T cur_diff = abs(input_vec[i] - output_S[i]);
      if (cur_diff > max_diff) max_diff = cur_diff;
      if ((cur_diff / abs(output_S[i])) > max_ratio)
        max_ratio = cur_diff / abs(output_S[i]);
    }
    S_error = max_diff;
    S_error_r = max_ratio;
    return max_diff;
  }

  T check_USV(std::vector<T> flat_A, std::vector<T> flat_U,
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
    float max_ratio = 0.0;
    for (int i = 0; i < (rows_A * cols_A); i++) {
      T cur_diff = abs(USV[i] - flat_A[i]);
      if (cur_diff > max_diff) max_diff = cur_diff;
      if ((cur_diff / abs(flat_A[i])) > max_ratio)
        max_ratio = cur_diff / abs(flat_A[i]);
    }
    A_error = max_diff;
    A_error_r = max_ratio;
    return max_diff;
  }

  T check_orthogonal(std::vector<T> flat_mat, unsigned rows, unsigned cols) {
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
  T run_test(sycl::queue q, int benchmark_rep = 1, bool print_matrices = false) {
    std::vector<T> flat_A = col_major_A();
    std::vector<T> flat_U(rows_A * rows_A);
    std::vector<T> flat_S(rows_A * cols_A);
    std::vector<T> flat_V(cols_A * cols_A);
    std::vector<ac_int<1, false>> rank_deficient(1);

    std::cout << "Running SVD test with input size " << rows_A << " x "
              << cols_A << ", repeating " << benchmark_rep << " time(s)"
              << std::endl;

    delta_time =
        singularValueDecomposition<rows_A, cols_A, k_fixed_iteration,
                                   k_raw_latency, k_zero_threshold_1e, T>(
            flat_A, flat_U, flat_S, flat_V, rank_deficient, q, benchmark_rep);

    throughput = (benchmark_rep / delta_time);

    compare_S(extract_singular_value(flat_S));

    A_error = check_USV(flat_A, flat_U, flat_S, flat_V);
    U_orthogonal_error = check_orthogonal(flat_U, rows_A, rows_A);
    V_orthogonal_error = check_orthogonal(flat_V, cols_A, cols_A);

    if (print_matrices) {
      std::cout << "S:\n";
      // print_matrix(extract_singular_value(flat_S), 1, rows_A);
      svd_testbench_tool::print_matrix<T>(flat_S, rows_A, cols_A, true);
      std::cout << "V:\n";
      svd_testbench_tool::print_matrix<T>(flat_V, cols_A, cols_A, true);
      std::cout << "U:\n";
      svd_testbench_tool::print_matrix<T>(flat_U, rows_A, rows_A, true);
    }
    return std::max({S_error, A_error, U_orthogonal_error, V_orthogonal_error});
  }

  void print_result() {
    std::cout << "Singular value error: " << S_error << "(" << S_error_r * 100
              << "%)" << std::endl;
    std::cout << "Decomposition error (A = USVt): " << A_error << "("
              << A_error_r * 100 << "%)" << std::endl;
    std::cout << "U orthogonal error: " << U_orthogonal_error << std::endl;
    std::cout << "V orthogonal error: " << V_orthogonal_error << std::endl;
    std::cout << "Total duration: " << delta_time << "s" << std::endl;
    std::cout << "Throughput: " << throughput * 1e-3 << "k matrices/s"
              << std::endl;
  }
};

SVDTestcase<float, 4, 4> small_4x4(
    std::vector<std::vector<float>>{
        {0.47084338, 0.99594452, 0.47982739, 0.69202168},
        {0.45148837, 0.72836647, 0.64691844, 0.62442883},
        {0.80974833, 0.82555856, 0.30709051, 0.58230306},
        {0.97898197, 0.98520343, 0.40133633, 0.85319924}},
    std::vector<float>({2.79495619, 0.44521050, 0.19458290, 0.07948970}));

SVDTestcase<float, 16, 8> testcase_16x8(
    std::vector<std::vector<float>>{
        {0.09197247, 0.14481869, 0.10407299, 0.25374938, 0.47811572, 0.63954233,
         0.04104508, 0.38657333},
        {0.92316611, 0.76245709, 0.61539652, 0.89160593, 0.77919693, 0.14006746,
         0.64050778, 0.88513825},
        {0.88713833, 0.21569021, 0.52698917, 0.25837260, 0.62761090, 0.16705069,
         0.55006137, 0.15100562},
        {0.17933577, 0.44237509, 0.29164377, 0.04858151, 0.14284620, 0.97584930,
         0.95781132, 0.97861833},
        {0.22954940, 0.53285279, 0.82211794, 0.24442794, 0.72065117, 0.82616029,
         0.82302578, 0.31588218},
        {0.52760637, 0.83106858, 0.68334733, 0.58536486, 0.10177759, 0.83382267,
         0.48252385, 0.33405913},
        {0.64459388, 0.44615274, 0.13607273, 0.84666874, 0.70038514, 0.05981429,
         0.68471502, 0.02031992},
        {0.19211154, 0.97691734, 0.21380459, 0.18721380, 0.33669170, 0.05466270,
         0.56268200, 0.05253976},
        {0.89958544, 0.17120118, 0.99595207, 0.38795272, 0.13999617, 0.22699871,
         0.28511385, 0.29012966},
        {0.70594215, 0.04854467, 0.21545484, 0.15641926, 0.43467411, 0.92386666,
         0.96494161, 0.19284229},
        {0.81370076, 0.90629365, 0.56153730, 0.26047083, 0.66264490, 0.83971270,
         0.61051658, 0.68128875},
        {0.76390120, 0.74742154, 0.83273900, 0.83469578, 0.21863598, 0.52614912,
         0.29617421, 0.87313192},
        {0.71767589, 0.42840114, 0.70372481, 0.82935507, 0.34454722, 0.92729788,
         0.30406199, 0.92858277},
        {0.99486099, 0.60156528, 0.30723120, 0.68557917, 0.29556701, 0.23800143,
         0.03078199, 0.19057876},
        {0.74059190, 0.68368920, 0.60495242, 0.66351287, 0.02082209, 0.90596643,
         0.79826228, 0.13455221},
        {0.99834564, 0.98115456, 0.26081567, 0.80371092, 0.57020481, 0.80252733,
         0.42442830, 0.54069138}},
    std::vector<float>({6.07216041, 1.71478328, 1.33994999, 1.14303961,
                        1.03316111, 0.94044096, 0.70064505, 0.65932612}));

#endif  // __SVD_TESTCASE__