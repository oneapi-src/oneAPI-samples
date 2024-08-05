#include <math.h>

#include <sycl/sycl.hpp>
#include <list>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"
// included from ../../../../include
#include "cholesky_inversion.hpp"

// Use "#define DEBUG" to print debugging information such as matrices content

/*
  COMPLEX, MATRIX_DIMENSION, FIXED_ITERATIONS_DECOMPOSITION and
  FIXED_ITERATIONS_INVERSION are defined by the build system. Depending on the
  value of COMPLEX, computes the real or complex Cholesky-based inversion. The
  Cholesky decompostion provides the L matrix from A such that: A = LL*
  Therefore we can compute inv(A) = inv(LL*)
                                  = inv(L*) x inv(L)
                                  = inv(L)* x inv(L)

  Function arguments:
  - a_matrix:    The input matrix.
  - i_matrix     The inverse matrix. The function will overwrite this matrix.
  - q:           The device queue.
  - matrix_count: Number of matrices to invert.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
template <typename T, bool is_complex>
void CholeskyInversion(std::vector<T> &a_matrix, std::vector<T> &i_matrix,
                       sycl::queue &q, int matrix_count, int repetitions) {
  CholeskyInversionImpl<MATRIX_DIMENSION, FIXED_ITERATIONS_DECOMPOSITION,
                        FIXED_ITERATIONS_INVERSION, is_complex, float>(
      a_matrix, i_matrix, q, matrix_count, repetitions);
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

/*
  Generate the input matrices for the cholesky inversion
*/
template <int matrices_to_invert, int matrix_size, int columns, typename T>
void GenerateInputData(std::vector<T> &a_matrix) {
  constexpr bool kComplex = COMPLEX != 0;

  std::cout << "Generating " << matrices_to_invert << " random ";
  if constexpr (kComplex) {
    std::cout << "complex ";
  } else {
    std::cout << "real ";
  }
  std::cout << "matri" << (matrices_to_invert > 1 ? "ces" : "x") << " of size "
            << columns << "x" << columns << " " << std::endl;
            
  constexpr size_t kRandomSeed = 1138;

  // Generate the random (Hermitian and positive-definite) input matrices
  srand(kRandomSeed);

  // Max condition number
  constexpr float kEpsilon = 0.5;

  // Random min and max values for the random floating-point value
  // generation
  constexpr float kRandomMin = 0;
  constexpr float kRandomMax = 1;

  /*
    Generate a random matrix with a given epsilon such that
    cond(M, inf) <= (1+epsilon)/(1-epsilon)
    This is helpful as having a condition number with infinite norm close to 1
    improves the numerical stability of the matrix inversion.
    Provided an epsilon value, this function populates the output vector with
    a matrix in a row fashion.

    Algorithm courtesy of Carl Christian Kjelgaard Mikkelsen (spock@cs.umu.se)

    Once this matrix is generated, we need to alter it a little to make
    it Hermitian
  */
  for (int mat_idx = 0; mat_idx < matrices_to_invert; mat_idx++) {
    int current_matrix = mat_idx * matrix_size;

    // Generate a random matrix R with diagonal elements set to 0
    // and measure the weights of the off diagonal entries
    std::vector<T> r, weights, a_matrix_non_hermitian;
    r.resize(columns * columns);
    a_matrix_non_hermitian.resize(columns * columns);
    weights.resize(columns);
    for (int row = 0; row < columns; row++) {
      weights[row] = {0};
      for (int col = 0; col < columns; col++) {
        if (col != row) {
          int index = (row * columns) + col;
          float random1 = RandomValueInInterval(kRandomMin, kRandomMax);
          T elem;
#if COMPLEX == 1
          float random1I = RandomValueInInterval(kRandomMin, kRandomMax);
          elem = {random1, random1I};
          r[index] = elem;
#else
          elem = random1;
          r[index] = elem;
#endif
          weights[row] += elem;
        }
      }

      // Construct the new diagonal element
      weights[row] /= kEpsilon;
      r[(row * columns) + row] = weights[row];
    }

    // Perform the diagonal scaling by solving:
    // diag(diag(A))*output = A
    for (int row = 0; row < columns; row++) {
      for (int col = 0; col < columns; col++) {
        int index = row * columns + col;
        a_matrix_non_hermitian[index] = r[index] / r[(row * columns) + row];
      }
    }

    // Make the matrix Hermitian
    for (int row = 0; row < columns; row++) {
      for (int col = 0; col < columns; col++) {
        int index = row * columns + col;

        a_matrix[current_matrix + index] =
            (a_matrix_non_hermitian[index] +
             a_matrix_non_hermitian[(col * columns) + row]) /
            2;

#if COMPLEX == 1
        if (row > col) {
          a_matrix[current_matrix + index] =
              a_matrix[current_matrix + index].conj();
        }
#endif
      }
    }

#ifdef DEBUG
    std::cout << "A MATRIX " << mat_idx << std::endl;
    for (size_t row = 0; row < columns; row++) {
      for (size_t col = 0; col < columns; col++) {
        std::cout << a_matrix[current_matrix + (col * columns) + row] << " ";
      }  // end of col
      std::cout << std::endl;
    }  // end of row
#endif

  }  // end of mat_idx
}

/*
  Check results for correctness
*/
template <int inverted_matrices, int matrix_size, int invert_matrix_size,
          int rows, int columns, typename T>
int CheckResults(std::vector<T> &a_matrix, std::vector<T> &i_matrix) {
  // For output post-processing (op)
  T i_matrix_op[rows][columns];

  // Floating-point error threshold value at which we decide that the design
  // computed an incorrect value
  constexpr float kErrorThreshold = 1e-4;

  // Check I matrices
  std::cout << "Verifying results..." << std::endl;
  for (int mat_idx = 0; mat_idx < inverted_matrices; mat_idx++) {
    // Keep track of I element index
    size_t i_idx = 0;

    // Read the I matrix from the output vector to the i_matrix_op matrix
    for (size_t j = 0; j < columns; j++) {
      for (size_t i = 0; i < rows; i++) {
        if (i < j) {
#if COMPLEX == 0
          i_matrix_op[i][j] = i_matrix_op[j][i];
#else
          i_matrix_op[i][j] = i_matrix_op[j][i].conj();
#endif
        } else {
          i_matrix_op[i][j] = i_matrix[(mat_idx * invert_matrix_size) + i_idx];
          i_idx++;
        }
      }
    }

#ifdef DEBUG
    std::cout << "I MATRIX" << std::endl;
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        std::cout << i_matrix_op[i][j] << " ";
      }
      std::cout << std::endl;
    }
#endif

    // Count the number of errors found for this matrix
    size_t error_count = 0;
    bool error = false;

    // Current A matrix start index
    int current_matrix = mat_idx * matrix_size;

    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < columns; j++) {
        // Compute I x A at index i,j
        T i_times_a_ij{0};
        // Compute A x I at index i,j
        T a_times_i_ij{0};

        for (size_t k = 0; k < columns; k++) {
#if COMPLEX == 0
          i_times_a_ij +=
              i_matrix_op[i][k] * a_matrix[current_matrix + (k * columns) + j];
          a_times_i_ij +=
              a_matrix[current_matrix + (i * columns) + k] * i_matrix_op[k][j];
#else
          i_times_a_ij += i_matrix_op[i][k] *
                          a_matrix[current_matrix + (k * columns) + j].conj();
          a_times_i_ij += a_matrix[current_matrix + (i * columns) + k] *
                          i_matrix_op[k][j].conj();
#endif
        }

        // Verify that all the results are OK:
        // I x A = Id at index i,j
        bool i_times_a_is_id = false;
        // A x I = Id at index i,j
        bool a_times_i_is_id = false;
        // I is finite at index i,j
        bool i_is_finite = false;

#if COMPLEX == 0
        if (i == j) {
          // Diagonal elements
          i_times_a_is_id = (abs(i_times_a_ij - 1)) < kErrorThreshold;
          a_times_i_is_id = (abs(a_times_i_ij - 1)) < kErrorThreshold;
        } else {
          // Non diagonal elements
          i_times_a_is_id = abs(i_times_a_ij) < kErrorThreshold;
          a_times_i_is_id = abs(a_times_i_ij) < kErrorThreshold;
        }
#else
        if (i == j) {
          // Diagonal elements
          i_times_a_is_id = (abs(i_times_a_ij.r() - 1)) < kErrorThreshold;
          a_times_i_is_id = (abs(a_times_i_ij.r() - 1)) < kErrorThreshold;
        } else {
          // Non diagonal elements
          i_times_a_is_id = abs(i_times_a_ij.r()) < kErrorThreshold;
          a_times_i_is_id = abs(a_times_i_ij.r()) < kErrorThreshold;
        }

        bool imag_is_zero_i_times_a = abs(i_times_a_ij.i()) < kErrorThreshold;
        i_times_a_is_id &= imag_is_zero_i_times_a;
        bool imag_is_zero_a_times_i = abs(a_times_i_ij.i()) < kErrorThreshold;
        a_times_i_is_id &= imag_is_zero_a_times_i;
#endif

        i_is_finite = IsFinite(i_matrix_op[i][j]);

        // If any of the checks failed
        if (!i_times_a_is_id || !a_times_i_is_id || !i_is_finite) {
          // Increase the error count for this matrix
          error_count++;

          // Continue counting the errors even if we are going to
          // produce an error
          if (error) {
            continue;
          }

          std::cerr << "Error in matrix " << mat_idx << std::endl;

          if (!i_times_a_is_id) {
            std::cerr << "Error: I*A at [" << i << "][" << j
                      << "] = " << i_times_a_ij << std::endl;
          }
          if (!a_times_i_is_id) {
            std::cerr << "Error: A*I at [" << i << "][" << j
                      << "] = " << a_times_i_ij << std::endl;
          }
          if (!i_is_finite) {
            std::cerr << "I[" << i << "][" << j << "] = " << i_matrix_op[i][j]
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

  // All passed
  std::cout << std::endl << "PASSED" << std::endl;
  return 0;
}

int main(int argc, char *argv[]) {
  constexpr size_t kRows = MATRIX_DIMENSION;
  constexpr size_t kColumns = MATRIX_DIMENSION;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kIMatrixSize = kColumns * (kColumns + 1) / 2;
  constexpr bool kComplex = COMPLEX != 0;

#if defined(FPGA_SIMULATOR)
  constexpr size_t kMatricesToInvert = 1;
#else
  constexpr size_t kMatricesToInvert = 8;
#endif

  // Get the number of times we want to repeat the inversion from the command
  // line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#elif defined(FPGA_SIMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 819200;
#endif

  if (repetitions < 1) {
    std::cerr << "Number of repetitions given is lower than 1." << std::endl;
    std::cerr << "The inversion must occur at least 1 time." << std::endl;
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
    std::vector<T> i_matrix;
    std::vector<T> r, weights;

    a_matrix.resize(kAMatrixSize * kMatricesToInvert);
    i_matrix.resize(kIMatrixSize * kMatricesToInvert);

    // Generate the input matrices to be inverted
    GenerateInputData<kMatricesToInvert, kAMatrixSize, kColumns>(a_matrix);

    std::cout << "Computing the Cholesky-based inversion of "
              << kMatricesToInvert << " matri"
              << (kMatricesToInvert > 1 ? "ces " : "x ") << repetitions
              << " times" << std::endl;

    // Invert the matrices
    CholeskyInversion<T, kComplex>(a_matrix, i_matrix, q, kMatricesToInvert,
                                   repetitions);

    // Check the returned matrices for correctness
    return CheckResults<kMatricesToInvert, kAMatrixSize, kIMatrixSize, kRows,
                        kColumns>(a_matrix, i_matrix);

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
              << ((kAMatrixSize + kIMatrixSize) * 2 * kMatricesToInvert *
                  sizeof(float)) /
                     pow(2, 30)
              << " GBs of memory was requested for the inversion of a "
              << "matrix of size " << kRows << " x " << kColumns << std::endl;
    std::terminate();
  }
}  // end of main
