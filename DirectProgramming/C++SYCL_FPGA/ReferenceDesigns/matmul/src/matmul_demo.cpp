#include <iomanip>
#include <iostream>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "matmul.hpp"

// Fills a matrix with random numbers within the range [lbound, ubound).
void FillRand(float *M_matrix, int lbound, int ubound, int elements) {
  for (int element = 0; element < elements; element++) {
    M_matrix[element] =
        static_cast<float>(rand()) /
            (static_cast<float>((RAND_MAX) / (ubound - lbound))) +
        lbound;
  }
}

// Compares two matrices; returns true iff they are equal.
bool EqualMat(float *C_matrix, float *C_reference, int rows, int cols,
              int kNumMatrices) {
  int matsize = rows * cols;
  bool passed = true;
  constexpr float kEpsilon = 0.01f;  // floating-point error threshold value

  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        int idx = matrix_idx * matsize + col * rows + row;
        if (abs(C_matrix[idx] - C_reference[idx]) > kEpsilon) {
          passed = false;
#if DEBUG == 1
          std::cout << "Error: C[" << col << "][" << row << "] = "
                    << C_matrix[idx] 
                    << " but REF[" << col << "][" << row << "] = "
                    << C_reference[idx] << std::endl;
#endif
        }
        if (!std::isfinite(C_matrix[idx])) {
          passed = false;
#if DEBUG == 1
          std::cout << "C[" << col << "][" << row << "] = " << C_matrix[idx]
                    << " is not finite" << std::endl;
#endif
        }
      }
    }
  }
  return passed;
}

// Output a matrix to the screen (assumes column-major format).
void PrintMat(float *M_matrix, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      std::cout << std::fixed;
      std::cout << std::setprecision(2);
      std::cout << std::setw(8) << M_matrix[col * rows + row] << " ";
    }
    std::cout << std::endl;
  }
}

// Transpose given matrices in m and store the result in m_transposed.
void TransposeMat(float *M_matrix, float *M_transposed, int rows, int cols,
                  int kNumMatrices) {
  int matsize = rows * cols;

  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        M_transposed[matrix_idx * matsize + row * cols + col] =
            M_matrix[matrix_idx * matsize + col * rows + row];
      }
    }
  }
}

// Multiply two matrices and store to a third matrix.
void MatmulRef(float *A_matrix, float *B_matrix, float *C_matrix, int rows_A,
               int common, int cols_B, int kNumMatrices) {
  int matsize_A = rows_A * common;
  int matsize_B = cols_B * common;
  int matsize_C = rows_A * cols_B;

  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    for (int col = 0; col < cols_B; col++) {
      for (int row = 0; row < rows_A; row++) {
        float sum = 0;
        for (int k = 0; k < common; k++) {
          sum += A_matrix[matrix_idx * matsize_A + k * rows_A + row] *
                 B_matrix[matrix_idx * matsize_B + col * common + k];
        }
        C_matrix[matrix_idx * matsize_C + col * rows_A + row] = sum;
      }
    }
  }
}

int main(int argc, char *argv[]) {

  // Enable the queue profiling to time the execution
  sycl::property_list queue_properties{
      sycl::property::queue::enable_profiling()};

  // Choose a selector that was selected by the default FPGA build system
#if FPGA_SIMULATOR
  sycl::queue q(sycl::ext::intel::fpga_simulator_selector_v,
                fpga_tools::exception_handler, queue_properties);
#elif FPGA_HARDWARE
  sycl::queue q(sycl::ext::intel::fpga_selector_v,
                fpga_tools::exception_handler, queue_properties);
#else  // FPGA_EMULATOR
  sycl::queue q(sycl::ext::intel::fpga_emulator_selector_v,
                fpga_tools::exception_handler, queue_properties);
#endif

  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>().c_str()
            << std::endl;

  // Repetitions and number of matrices to measure performance
#if FPGA_SIMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
  int kNumMatrices = 1;
#elif FPGA_HARDWARE
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
  int kNumMatrices = 8;
#else  // FPGA_EMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
  int kNumMatrices = 8;
#endif

  // Matrix sizes
  constexpr int kMatsizeA = ROWS_A * COMMON;
  constexpr int kMatsizeB = COLS_B * COMMON;
  constexpr int kMatsizeC = ROWS_A * COLS_B;

  // Create arrays to hold the input and output matrices
  float A_matrix[kMatsizeA * kNumMatrices];
  float B_matrix[kMatsizeB * kNumMatrices];
  float C_matrix[kMatsizeC * kNumMatrices];

  // Generate random A and B matrices
  constexpr int kRandMin = 1;
  constexpr int kRandMax = 10;
  srand(1138);
  FillRand(A_matrix, kRandMin, kRandMax, kMatsizeA * kNumMatrices);
  FillRand(B_matrix, kRandMin, kRandMax, kMatsizeB * kNumMatrices);

  // Calculate a reference to compare our answer to and store it in C_reference
  // NOTE: since the systolic matrix multiply interprets B as transposed, we
  // need to first transpose B_matrix to B_transposed to use it in the standard
  // MM algorithm
  float B_transposed[kMatsizeB * kNumMatrices];
  float C_reference[kMatsizeC * kNumMatrices];
  TransposeMat(B_matrix, B_transposed, COLS_B, COMMON, kNumMatrices);
  MatmulRef(A_matrix, B_transposed, C_reference, ROWS_A, COMMON, COLS_B,
             kNumMatrices);

  // Run the matrix multiplication
  std::cout << " Matrix A size: " << ROWS_A << " x " << COMMON
            << " (tile: " << TILE_A << " x " << TILE_COMMON << ")" << std::endl
            << " Matrix B size: " << COMMON << " x " << COLS_B
            << " (tile: " << TILE_COMMON << " x " << TILE_B << ")" << std::endl
            << " Systolic array size: " << TILE_A << " x " << TILE_B << " PEs"
            << std::endl;
  std::cout << "Running matrix multiplication of " << kNumMatrices
            << ((kNumMatrices > 1) ? " matrices " : " matrix ") << repetitions
            << " times" << std::endl;

  MATMULImpl<float, ROWS_A, COMMON, COLS_B, TILE_A, TILE_B, TILE_COMMON>(
      q, A_matrix, B_matrix, C_matrix, repetitions, kNumMatrices);

#if DEBUG == 1
  // Print A, B, C and reference matrices
  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    std::cout << std::endl << matrix_idx << std::endl;
    std::cout << std::endl << "Matrix A" << std::endl;
    PrintMat(A_matrix + matrix_idx * kMatsizeA, ROWS_A, COMMON);
    std::cout << std::endl << "Matrix B" << std::endl;
    PrintMat(B_transposed + matrix_idx * kMatsizeB, COMMON, COLS_B);
    std::cout << std::endl << "Matrix C reference" << std::endl;
    PrintMat(C_reference + matrix_idx * kMatsizeC, ROWS_A, COLS_B);
    std::cout << std::endl << "Matrix C calculated" << std::endl;
    PrintMat(C_matrix + matrix_idx * kMatsizeC, ROWS_A, COLS_B);
  }
#endif

  // Verify results
  bool passed = EqualMat(C_matrix, C_reference, ROWS_A, COLS_B, kNumMatrices);
  std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;

  return !passed;
}