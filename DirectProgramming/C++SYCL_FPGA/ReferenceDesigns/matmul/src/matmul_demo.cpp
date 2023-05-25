#include <iomanip>
#include <iostream>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"
#include "matmul.hpp"

// Fills a matrix with random numbers within the range [l_bound, u_bound).
void FillRand(std::vector<float> &m_matrix, int l_bound, int u_bound,
              int elements) {
  for (int element = 0; element < elements; element++) {
    m_matrix[element] =
        static_cast<float>(rand()) /
            (static_cast<float>((RAND_MAX) / (u_bound - l_bound))) +
        l_bound;
  }
}

// Compares num_matrices pairs of matrices; returns true iff they are equal
// given a tolerated error bound.
bool EqualMat(std::vector<float> &c_matrix, std::vector<float> &c_reference,
              int rows, int cols, int num_matrices) {
  int matsize = rows * cols;
  bool passed = true;

  // Floating-point error threshold value
  constexpr float kEpsilon = 0.01f;

  for (int matrix_idx = 0; matrix_idx < num_matrices; matrix_idx++) {
    for (int col = 0; col < cols; col++) {
      for (int row = 0; row < rows; row++) {
        int idx = matrix_idx * matsize + col * rows + row;
        if (abs(c_matrix[idx] - c_reference[idx]) > kEpsilon) {
          passed = false;
#if DEBUG
          std::cout << "Error: C[" << col << "][" << row << "] = "
                    << c_matrix[idx]
                    << " but REF[" << col << "][" << row << "] = "
                    << c_reference[idx] << std::endl;
#endif
        }
        if (!std::isfinite(c_matrix[idx])) {
          passed = false;
#if DEBUG
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
void PrintMat(std::vector<float> &m_matrix, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      // Copy old state of cout
      std::ios oldState(nullptr);
      oldState.copyfmt(std::cout);

      // Edit the output format of cout
      std::cout << std::fixed << std::setprecision(2);

      // Print the results
      std::cout << std::setw(8) << m_matrix[col * rows + row] << " ";

      // Restore the output format of cout
      std::cout.copyfmt(oldState);
    }
    std::cout << std::endl;
  }
}

// Transpose num_matrices matrices in m_matrix and store the results in
// m_transposed.
void TransposeMat(std::vector<float> &m_matrix,
                  std::vector<float> &m_transposed, int rows, int cols,
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

// Multiply num_matrices pairs of matrices from a_matrix and b_matrix and store
// all the results in c_matrix.
void MatmulRef(std::vector<float> &a_matrix, std::vector<float> &b_matrix,
               std::vector<float> &c_matrix, int rows_a, int common, int cols_b,
               int num_matrices) {
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
  // Matrix paramters specified by build system
  constexpr int kRowsA = ROWS_A;
  constexpr int kCommon = COMMON;
  constexpr int kColsB = COLS_B;
  constexpr int kTileA = TILE_A;
  constexpr int kTileB = TILE_B;

  // Matrix sizes
  constexpr int kMatsizeA = kRowsA * kCommon;
  constexpr int kMatsizeB = kColsB * kCommon;
  constexpr int kMatsizeC = kRowsA * kColsB;

  // Repetitions and number of matrices to measure performance
#if FPGA_SIMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
  constexpr int kNumMatrices = 1;
#elif FPGA_HARDWARE
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
  constexpr int kNumMatrices = 2;
#else // #if FPGA_EMULATOR
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
  constexpr int kNumMatrices = 2;
#endif

  try {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
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

  // Create arrays to hold the input and output matrices
  std::vector<float> a_matrix(kMatsizeA * kNumMatrices);
  std::vector<float> b_matrix(kMatsizeB * kNumMatrices);
  std::vector<float> c_matrix(kMatsizeC * kNumMatrices);

  // Generate random A and B matrices
  constexpr int kRandMin = 1;
  constexpr int kRandMax = 10;
  srand(1138);
  FillRand(a_matrix, kRandMin, kRandMax, kMatsizeA * kNumMatrices);
  FillRand(b_matrix, kRandMin, kRandMax, kMatsizeB * kNumMatrices);

  // Calculate a reference to compare our answer to and store it in c_reference
  // NOTE: since the systolic matrix multiply interprets B as transposed, we
  // need to first transpose b_matrix to b_transposed to use it in the standard
  // MM algorithm
  std::vector<float> b_transposed(kMatsizeB * kNumMatrices);
  std::vector<float> c_reference(kMatsizeC * kNumMatrices);
  TransposeMat(b_matrix, b_transposed, kColsB, kCommon, kNumMatrices);
  MatmulRef(a_matrix, b_transposed, c_reference, kRowsA, kCommon, kColsB,
            kNumMatrices);

  std::cout << " Matrix A size: " << kRowsA << " x " << kCommon
            << " (tile: " << kTileA << " x " << kCommon << ")" << std::endl
            << " Matrix B size: " << kCommon << " x " << kColsB
            << " (tile: " << kCommon << " x " << kTileB << ")" << std::endl
            << " Systolic array size: " << kTileA << " x " << kTileB << " PEs"
            << std::endl;
  std::cout << "Running matrix multiplication of " << kNumMatrices
            << ((kNumMatrices > 1) ? " matrices " : " matrix ") << repetitions
            << " times" << std::endl;

  // Run the matrix multiplication
  MatmulImpl<float, kRowsA, kCommon, kColsB, kTileA, kTileB, kNumMatrices>(
      q, a_matrix, b_matrix, c_matrix, repetitions);

#if DEBUG
  // Print A, B, C and reference matrices
  for (int matrix_idx = 0; matrix_idx < kNumMatrices; matrix_idx++) {
    std::cout << std::endl << matrix_idx << std::endl;

    std::cout << std::endl << "Matrix A" << std::endl;
    std::vector<float> a_vector = {
        a_matrix.begin() + matrix_idx * kMatsizeA,
        a_matrix.begin() + (matrix_idx + 1) * kMatsizeA};
    PrintMat(a_vector, kRowsA, kCommon);

    std::cout << std::endl << "Matrix B" << std::endl;
    std::vector<float> b_vector = {
        b_transposed.begin() + matrix_idx * kMatsizeB,
        b_transposed.begin() + (matrix_idx + 1) * kMatsizeB};
    PrintMat(b_vector, kCommon, kColsB);

    std::cout << std::endl << "Matrix C reference" << std::endl;
    std::vector<float> c_ref_vector = {
        c_reference.begin() + matrix_idx * kMatsizeC,
        c_reference.begin() + (matrix_idx + 1) * kMatsizeC};
    PrintMat(c_ref_vector, kRowsA, kColsB);

    std::cout << std::endl << "Matrix C calculated" << std::endl;
    std::vector<float> c_vector = {
        c_matrix.begin() + matrix_idx * kMatsizeC,
        c_matrix.begin() + (matrix_idx + 1) * kMatsizeC};
    PrintMat(c_vector, kRowsA, kColsB);
  }
#endif

  // Verify results
  bool passed = EqualMat(c_matrix, c_reference, kRowsA, kColsB, kNumMatrices);
  std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;

  return !passed;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting an FPGA hardware, "
                 "ensure that your system is plugged to an FPGA board that is "
                 "set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  }
} // end of main