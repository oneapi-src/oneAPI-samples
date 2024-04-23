#include <math.h>

#include <sycl/sycl.hpp>
#include <list>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "cholesky.hpp"
#include "exception_handler.hpp"

// Use "#define DEBUG" to print debugging information such as matrices content

/*
  COMPLEX, MATRIX_DIMENSION and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex Cholesky decomposition
  is defined (A = LL*)

  Function arguments:
  - a_matrix:    The input matrix.
  - l_matrix     The L matrix. The function will overwrite this matrix.
                 The vector will only contain the lower triangular elements
                 of the matrix, in a row by row fashion.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
template <typename T, bool is_complex>
void CholeskyDecomposition(std::vector<T> &a_matrix, std::vector<T> &l_matrix,
                           sycl::queue &q, int matrix_count, int repetitions) {
  CholeskyDecompositionImpl<MATRIX_DIMENSION, FIXED_ITERATIONS, is_complex,
                            float>(a_matrix, l_matrix, q, matrix_count,
                                   repetitions);
}

/*
  Returns true if both the real and complex parts of the given ac_complex
  value are finite
*/
bool IsFinite(ac_complex<float> val) {
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  Returns true if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

/*
  Returns a random floating-point value between min and max
*/
float RandomValueInInterval(float min, float max) {
  return min + static_cast<float>(rand()) /
                   (static_cast<float>(RAND_MAX) / (max - min));
}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRows = MATRIX_DIMENSION;
  constexpr size_t kColumns = MATRIX_DIMENSION;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kLMatrixSize = (kColumns * (kColumns + 1)) / 2;
  constexpr bool kComplex = COMPLEX != 0;

#if defined(FPGA_SIMULATOR)
  constexpr size_t kMatricesToDecompose = 1;
#else
  constexpr size_t kMatricesToDecompose = 8;
#endif

  // Get the number of times we want to repeat the decomposition
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#elif defined(FPGA_SIMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#endif

  if (repetitions < 1) {
    std::cerr << "Number of repetitions given is lower than 1." << std::endl;
    std::cerr << "The decomposition must occur at least 1 time." << std::endl;
    std::cerr << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

  try {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

    // Enable the queue profiling to time the execution
    sycl::queue q = sycl::queue(
        selector, fpga_tools::exception_handler,
        sycl::property_list{sycl::property::queue::enable_profiling()});
    sycl::device device = q.get_device();

    std::cout << "Running on device: "
        << device.get_info<sycl::info::device::name>().c_str() 
        << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> l_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    l_matrix.resize(kLMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size " << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random (hermitian and positive-definite) input matrices
    srand(kRandomSeed);

    for (int mat_idx = 0; mat_idx < kMatricesToDecompose; mat_idx++) {
      // Construct a single random hermitian and positive-definite matrix
      // To do so we, we generate a hermitian matrix A where each element
      // is between 0 and 1.
      // Since A(i,j) < 1 by construction and a symmetric diagonally dominant
      // matrix is symmetric positive definite we can be sure to have a
      // symmetric diagonally dominant matrix by adding nI to A
      // A = A + n*eye(n);
      // For complex matrices, the diagonal elements must be real.

      // Random min and max values for the random floating-point value
      // generation
      constexpr float kRandomMin = 0;
      constexpr float kRandomMax = 1;

      int current_matrix = mat_idx * kAMatrixSize;

      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          float diag_scaling = (row == col) ? float{kRows} : 0;

          int index = current_matrix + (col * kRows) + row;
          int transpose_index = current_matrix + (row * kRows) + col;

          if (col >= row) {
            float random_real = RandomValueInInterval(kRandomMin, kRandomMax);
#if COMPLEX == 0
            a_matrix[index] = random_real + diag_scaling;
#else
            float random_imag =
                row == col ? float{0}
                           : RandomValueInInterval(kRandomMin, kRandomMax);
            ac_complex<float> random_complex{random_real + diag_scaling,
                                             random_imag};
            a_matrix[index] = random_complex;
#endif
          } else {
            // conjugate transpose
#if COMPLEX == 0
            a_matrix[index] = a_matrix[transpose_index];
#else
            a_matrix[index] = a_matrix[transpose_index].conj();
#endif
          }
        }  // end of col
      }    // end of row

#ifdef DEBUG
      std::cout << "A MATRIX " << mat_idx << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << a_matrix[current_matrix + (col * kRows) + row] << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
#endif

    }  // end of mat_idx

    std::cout << "Computing the Cholesky decomposition of "
              << kMatricesToDecompose << " matri"
              << (kMatricesToDecompose > 1 ? "ces " : "x ") << repetitions
              << " times" << std::endl;

    CholeskyDecomposition<T, kComplex>(a_matrix, l_matrix, q,
                                          kMatricesToDecompose, repetitions);

    // For output post-processing (op)
    T l_matrix_op[kRows][kColumns];

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;

    // Check L matrices
    std::cout << "Verifying results..." << std::endl;
    for (int mat_idx = 0; mat_idx < kMatricesToDecompose; mat_idx++) {
      // Keep track of L element index
      size_t l_idx = 0;

      // Read the L matrix from the output vector to the l_matrix_op matrix
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          if (j > i)
            l_matrix_op[i][j] = 0;
          else {
            l_matrix_op[i][j] = l_matrix[(mat_idx * kLMatrixSize) + l_idx];
            l_idx++;
          }
        }
      }

#ifdef DEBUG
      std::cout << "L MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << l_matrix_op[i][j] << " ";
        }
        std::cout << std::endl;
      }
#endif

      // Count the number of errors found for this matrix
      size_t error_count = 0;
      bool error = false;

      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          // Compute LL* at index i,j
          T l_l_star_ij{0};
          for (size_t k = 0; k < kColumns; k++) {
#if COMPLEX == 0
            l_l_star_ij += (l_matrix_op[i][k] * l_matrix_op[j][k]);
#else
            l_l_star_ij += (l_matrix_op[i][k] * l_matrix_op[j][k].conj());
#endif
          }

          // Verify that all the results are OK:
          // LL* = A at index i,j
          bool ll_star_eq_a;
          // L is finite at index i,j
          bool l_is_finite;

          int current_matrix = mat_idx * kAMatrixSize;
          int current_element = (j * kRows) + i;

#if COMPLEX == 0
          ll_star_eq_a = abs(a_matrix[current_matrix + current_element] -
                             l_l_star_ij) < kErrorThreshold;

#else
          ll_star_eq_a = (abs(a_matrix[current_matrix + current_element].r() -
                              l_l_star_ij.r()) < kErrorThreshold) &&
                         (abs(a_matrix[current_matrix + current_element].i() -
                              l_l_star_ij.i()) < kErrorThreshold);
#endif

          l_is_finite = ((i < kColumns) && IsFinite(l_matrix_op[i][j])) ||
                        (i >= kColumns);

          // If any of the checks failed
          if (!ll_star_eq_a || !l_is_finite) {
            // Increase the error count for this matrix
            error_count++;

            // Continue counting the errors even if we are going to
            // produce an error
            if (error) {
              continue;
            }

            std::cerr << "Error in matrix " << mat_idx << std::endl;

            if (!ll_star_eq_a) {
              std::cerr
                  << "Error: A[" << i << "][" << j << "] = "
                  << a_matrix[(current_matrix * kAMatrixSize) + (j * kRows) + i]
                  << " but LL*[" << i << "][" << j << "] = " << l_l_star_ij
                  << std::endl;
            }
            if (!l_is_finite) {
              std::cerr << "L[" << i << "][" << j << "] = " << l_matrix_op[i][j]
                        << " is not finite" << std::endl;
            }
            error = true;
          }
        }  // end of j
      }    // end of i

      if (error_count > 0) {
        std::cerr << std::endl << "FAILED" << std::endl;
        std::cerr << std::endl
                  << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
        return 1;
      }
    }  // end of mat_idx

    std::cout << std::endl << "PASSED" << std::endl;
    return 0;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what()
              << std::endl;
    std::cerr << "   If you are targeting FPGA hardware, "
                 "ensure that your system is connected to an FPGA board that "
                 "is set up correctly"
              << std::endl;
    std::cerr << "   If you are targeting the FPGA emulator, compile with "
                 "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: "
              << e.what() << std::endl;
    std::cerr << "   You can reduce the memory requirement by reducing the "
                 "number of matrices generated. Specify a smaller number when "
                 "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << ((kAMatrixSize + kLMatrixSize) * 2 * kMatricesToDecompose *
                  sizeof(float)) /
                     pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kRows << " x " << kColumns << std::endl;
    std::terminate();
  }
}  // end of main
