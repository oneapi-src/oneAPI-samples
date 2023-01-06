#include <iomanip>
#include <iostream>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"
#include "matmul.hpp"

#ifndef OVERRIDE_PARAMS
  #define ROWS_A 8
  #define COMMON 64
  #define COLS_B 8
  #define TILE_A 8
  #define TILE_B 8
  #define TILE_COMMON 64
#endif

using namespace sycl;
using namespace fpga_tools;

// Fills a matrix with random numbers within the range [lbound, ubound).
void fill_rand(float *M, int lbound, int ubound, int elements) {
  for (int element = 0; element < elements; element++) {
    M[element] = static_cast<float>(rand()) /
                     (static_cast<float>((RAND_MAX) / (ubound - lbound))) +
                 lbound;
  }
}

// Compares two matrices; returns true iff they are equal.
bool equal_mat(float *MC, float *MREF, int rows, int cols, int num_matrices) {
  int matsize = rows * cols;
  bool passed = true;
  constexpr float epsilon = 0.01f;  // floating-point error threshold value

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        int idx = matrix_idx * matsize + col * rows + row;
        if (abs(MC[idx] - MREF[idx]) > epsilon) {
          passed = false;
#if DEBUG == 1
          std::cout << "Error: C[" << col << "][" << row << "] = " << MC[idx]
                    << " but REF[" << col << "][" << row << "] = " << MREF[idx]
                    << std::endl;
#endif
        }
        if (!std::isfinite(MC[idx])) {
          passed = false;
#if DEBUG == 1
          std::cout << "C[" << col << "][" << row << "] = " << MC[idx]
                    << " is not finite" << std::endl;
#endif
        }
      }
    }
  }
  return passed;
}

// Output a matrix to the screen (assumes column-major format).
void print_mat(float *M, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      std::cout << std::fixed;
      std::cout << std::setprecision(2);
      std::cout << std::setw(8) << M[col * rows + row] << " ";
    }
    std::cout << std::endl;
  }
}

// Transpose given matrices in M and store the result in MT.
void transpose_mat(float *M, float *MT, int rows, int cols, int num_matrices) {
  int matsize = rows * cols;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        MT[matrix_idx * matsize + row * cols + col] =
            M[matrix_idx * matsize + col * rows + row];
      }
    }
  }
}

// Multiply two matrices and store to a third matrix.
void matmul_ref(float *A, float *B, float *C, int rows_A, int common,
                int cols_B, int num_matrices) {
  int matsize_A = rows_A * common;
  int matsize_B = cols_B * common;
  int matsize_C = rows_A * cols_B;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols_B; col++) {
      for (int row = 0; row < rows_A; row++) {
        float sum = 0;
        for (int k = 0; k < common; k++) {
          sum += A[matrix_idx * matsize_A + k * rows_A + row] *
                 B[matrix_idx * matsize_B + col * common + k];
        }
        C[matrix_idx * matsize_C + col * rows_A + row] = sum;
      }
    }
  }
}

int main(int argc, char *argv[]) {

  // SYCL boilerplate
#if FPGA_SIMULATOR
  ext::intel::fpga_simulator_selector device_selector;
#elif FPGA_HARDWARE
  ext::intel::fpga_selector device_selector;
#else // FPGA_EMULATOR
  ext::intel::fpga_emulator_selector device_selector;
#endif

  // Enable the queue profiling to time the execution
  property_list queue_properties{property::queue::enable_profiling()};
  queue q = queue(device_selector, exception_handler, queue_properties);
  std::cout << "Device: "
            << q.get_device().get_info<info::device::name>().c_str()
            << std::endl;

  // Repetitions and number of matrices to measure performance
#if FPGA_SIMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#elif FPGA_HARDWARE
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#else // FPGA_EMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#endif
  constexpr int num_matrices = 8;

  // Matrix sizes
  constexpr int matsize_A = ROWS_A * COMMON;
  constexpr int matsize_B = COLS_B * COMMON;
  constexpr int matsize_C = ROWS_A * COLS_B;

  // Create arrays to hold the input and output matrices
  float A[matsize_A * num_matrices];
  float B[matsize_B * num_matrices];
  float C[matsize_C * num_matrices];

  // Generate random A and B matrices
  constexpr int rand_min = 1;
  constexpr int rand_max = 10;
  srand(1138);
  fill_rand(A, rand_min, rand_max, matsize_A * num_matrices);
  fill_rand(B, rand_min, rand_max, matsize_B * num_matrices);

  // Calculate a reference to compare our answer to and store it in CR
  // NOTE: since the systolic matrix multiply interprets B as transposed, we
  // need to first transpose B to BT to use it in the standard MM algorithm
  float BT[matsize_B * num_matrices];
  float CR[matsize_C * num_matrices];
  transpose_mat(B, BT, COLS_B, COMMON, num_matrices);
  matmul_ref(A, BT, CR, ROWS_A, COMMON, COLS_B, num_matrices);

  // Run the matrix multiplication
  std::cout << " Matrix A size: " << ROWS_A << " x " << COMMON
            << " (tile: " << TILE_A << " x " << TILE_COMMON << ")" << std::endl
            << " Matrix B size: " << COMMON << " x " << COLS_B
            << " (tile: " << TILE_COMMON << " x " << TILE_B << ")" << std::endl
            << " Systolic array size: " << TILE_A << " x " << TILE_B << " PEs"
            << std::endl;
  std::cout << "Running matrix multiplication of " << num_matrices
            << ((num_matrices > 1) ? " matrices " : " matrix ") << repetitions
            << " times" << std::endl;
            
  MATMULImpl<float, ROWS_A, COMMON, COLS_B, TILE_A, TILE_B, TILE_COMMON>(
      q, A, B, C, repetitions, num_matrices);

#if DEBUG
  // Print A, B, C and reference matrices
  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    std::cout << std::endl << matrix_idx << std::endl;
    std::cout << std::endl << "Matrix A" << std::endl;
    print_mat(A + matrix_idx * matsize_A, ROWS_A, COMMON);
    std::cout << std::endl << "Matrix B" << std::endl;
    print_mat(BT + matrix_idx * matsize_B, COMMON, COLS_B);
    std::cout << std::endl << "Matrix C reference" << std::endl;
    print_mat(CR + matrix_idx * matsize_C, ROWS_A, COLS_B);
    std::cout << std::endl << "Matrix C calculated" << std::endl;
    print_mat(C + matrix_idx * matsize_C, ROWS_A, COLS_B);
  }
#endif

  // Verify results
  bool passed = equal_mat(C, CR, ROWS_A, COLS_B, num_matrices);
  std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;
  
  return !passed;
}