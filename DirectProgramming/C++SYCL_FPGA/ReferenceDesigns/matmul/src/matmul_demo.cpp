#include <iomanip>
#include <iostream>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "matmul.hpp"

// Fills a matrix with random numbers within the range [l_bound, u_bound).
void FillRand(float *m_matrix, int l_bound, int u_bound, int elements) {
  for (int element = 0; element < elements; element++) {
    m_matrix[element] =
        static_cast<float>(rand()) /
            (static_cast<float>((RAND_MAX) / (u_bound - l_bound))) +
        l_bound;
  }
}

// Compares num_matrices pairs of matrices; returns true iff they are equal
// given a tolerated error bound.
bool EqualMat(float *c_matrix, float *c_transposed, int rows, int cols,
              int num_matrices) {
  int matsize = rows * cols;
  bool passed = true;

  // Floating-point error threshold value
  constexpr float kEpsilon = 0.01f;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        int idx = matrix_idx * matsize + col * rows + row;
        if (abs(c_matrix[idx] - c_transposed[idx]) > kEpsilon) {
          passed = false;
#if DEBUG == 1
          std::cout << "Error: C[" << col << "][" << row << "] = "
                    << c_matrix[idx]
                    << " but REF[" << col << "][" << row << "] = "
                    << c_transposed[idx] << std::endl;
#endif
        }
        if (!std::isfinite(c_matrix[idx])) {
          passed = false;
#if DEBUG == 1
          std::cout << "C[" << col << "][" << row << "] = " << c_matrix[idx]
                    << " is not finite" << std::endl;
#endif
        }
      }
    }
  }
  return passed;
}

// Output a matrix to the screen (assumes column-major format).
void PrintMat(float *m_matrix, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      std::cout << std::fixed;
      std::cout << std::setprecision(2);
      std::cout << std::setw(8) << m_matrix[col * rows + row] << " ";
    }
    std::cout << std::endl;
  }
}

// Transpose num_matrices matrices in m_matrix and store the results in
// m_transposed.
void TransposeMat(float *m_matrix, float *m_transposed, int rows, int cols,
                  int num_matrices) {
  int matsize = rows * cols;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        m_transposed[matrix_idx * matsize + row * cols + col] =
            m_matrix[matrix_idx * matsize + col * rows + row];
      }
    }
  }
}

// Multiply num_matrices pairs of matrices and store all the results in
// c_matrix.
void MatmulRef(float *a_matrix, float *b_matrix, float *c_matrix, int rows_a,
               int common, int cols_b, int num_matrices) {
  int matsize_a = rows_a * common;
  int matsize_b = cols_b * common;
  int matsize_c = rows_a * cols_b;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols_b; col++) {
      for (int row = 0; row < rows_a; row++) {
        float sum = 0;
        for (int k = 0; k < common; k++) {
          sum += a_matrix[matrix_idx * matsize_a + k * rows_a + row] *
                 b_matrix[matrix_idx * matsize_b + col * common + k];
        }
        c_matrix[matrix_idx * matsize_c + col * rows_a + row] = sum;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  // Enable the queue profiling to time the execution
  sycl::property_list queue_properties{
      sycl::property::queue::enable_profiling()};
  sycl::queue q =
      sycl::queue(selector, fpga_tools::exception_handler, queue_properties);

  std::cout << "Running on device: "
            << q.get_device().get_info<sycl::info::device::name>().c_str()
            << std::endl;

  // Repetitions and number of matrices to measure performance
#if FPGA_SIMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
  int num_matrices = 1;
#elif FPGA_HARDWARE
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
  int num_matrices = 8;
#else  // #if FPGA_EMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
  int num_matrices = 8;
#endif

  // Matrix paramters specified by build system
  constexpr int kRowsA = ROWS_A;
  constexpr int kCommon = COMMON;
  constexpr int kColsB = COLS_B;
  constexpr int kTileA = TILE_A;
  constexpr int kTileB = TILE_B;
  constexpr int kTileCommon = TILE_COMMON;

  // Matrix sizes
  constexpr int kMatsizeA = kRowsA * kCommon;
  constexpr int kMatsizeB = kColsB * kCommon;
  constexpr int kMatsizeC = kRowsA * kColsB;

  // Create arrays to hold the input and output matrices
  float a_matrix[kMatsizeA * num_matrices];
  float b_matrix[kMatsizeB * num_matrices];
  float c_matrix[kMatsizeC * num_matrices];

  // Generate random A and B matrices
  constexpr int kRandMin = 1;
  constexpr int kRandMax = 10;
  srand(1138);
  FillRand(a_matrix, kRandMin, kRandMax, kMatsizeA * num_matrices);
  FillRand(b_matrix, kRandMin, kRandMax, kMatsizeB * num_matrices);

  // Calculate a reference to compare our answer to and store it in c_transposed
  // NOTE: since the systolic matrix multiply interprets B as transposed, we
  // need to first transpose b_matrix to b_transposed to use it in the standard
  // MM algorithm
  float b_transposed[kMatsizeB * num_matrices];
  float c_transposed[kMatsizeC * num_matrices];
  TransposeMat(b_matrix, b_transposed, kColsB, kCommon, num_matrices);
  MatmulRef(a_matrix, b_transposed, c_transposed, kRowsA, kCommon, kColsB,
            num_matrices);

  // Run the matrix multiplication
  std::cout << " Matrix A size: " << kRowsA << " x " << kCommon
            << " (tile: " << kTileA << " x " << kTileCommon << ")" << std::endl
            << " Matrix B size: " << kCommon << " x " << kColsB
            << " (tile: " << kTileCommon << " x " << kTileB << ")" << std::endl
            << " Systolic array size: " << kTileA << " x " << kTileB << " PEs"
            << std::endl;
  std::cout << "Running matrix multiplication of " << num_matrices
            << ((num_matrices > 1) ? " matrices " : " matrix ") << repetitions
            << " times" << std::endl;

  MatmulImpl<float, kRowsA, kCommon, kColsB, kTileA, kTileB, kTileCommon>(
      q, a_matrix, b_matrix, c_matrix, repetitions, num_matrices);

#if DEBUG == 1
  // Print A, B, C and reference matrices
  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    std::cout << std::endl << matrix_idx << std::endl;
    std::cout << std::endl << "Matrix A" << std::endl;
    PrintMat(a_matrix + matrix_idx * kMatsizeA, kRowsA, kCommon);
    std::cout << std::endl << "Matrix B" << std::endl;
    PrintMat(b_transposed + matrix_idx * kMatsizeB, kCommon, kColsB);
    std::cout << std::endl << "Matrix C reference" << std::endl;
    PrintMat(c_transposed + matrix_idx * kMatsizeC, kRowsA, kColsB);
    std::cout << std::endl << "Matrix C calculated" << std::endl;
    PrintMat(c_matrix + matrix_idx * kMatsizeC, kRowsA, kColsB);
  }
#endif

  // Verify results
  bool passed = EqualMat(c_matrix, c_transposed, kRowsA, kColsB, num_matrices);
  std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;

  return !passed;
}