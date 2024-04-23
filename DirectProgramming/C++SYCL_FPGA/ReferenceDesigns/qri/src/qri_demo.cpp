#include <math.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <list>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#include "exception_handler.hpp"

#include "qri.hpp"

#ifdef FPGA_SIMULATOR
#define ROWS_COMPONENT_V 8
#define COLS_COMPONENT_V 8
#else
#define ROWS_COMPONENT_V ROWS_COMPONENT
#define COLS_COMPONENT_V COLS_COMPONENT
#endif

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS_QRD and
  FIXED_ITERATIONS_QRI are defined by the build system.
  Depending on the value of COMPLEX, the real or complex QR based matrix
  inversion function (QRI) is defined.

  Each matrix (input and output) are represented using vectors and are
  interpreted in a column fashion (transposed).

  Function arguments:
  - a_matrix:    The input matrix to be inverted.
                Interpreted as a transposed matrix.
  - inv_matrix:  The output matrix. The function will overwrite this matrix.
                Will contain the inverse of a_matrix.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the a_matrix
                vector.
  - repetitions: The number of repetitions of the computation to execute.
                (for performance evaluation)
*/
#if COMPLEX == 0
// Real single precision floating-point QR based inversion
void QRI(std::vector<float> &a_matrix, std::vector<float> &inv_matrix,
         sycl::queue &q, size_t matrices, size_t repetitions) {
  constexpr bool is_complex = false;
  QRIImpl<COLS_COMPONENT_V, ROWS_COMPONENT_V, FIXED_ITERATIONS_QRD,
           FIXED_ITERATIONS_QRI, is_complex, float>(a_matrix, inv_matrix, q,
                                                   matrices, repetitions);
}
#else
// Complex single precision floating-point QR based inversion
void QRI(std::vector<ac_complex<float> > &a_matrix,
         std::vector<ac_complex<float> > &inv_matrix, sycl::queue &q,
         size_t matrices, size_t repetitions) {
  constexpr bool is_complex = true;
  QRIImpl<COLS_COMPONENT_V, ROWS_COMPONENT_V, FIXED_ITERATIONS_QRD,
           FIXED_ITERATIONS_QRI, is_complex, float>(a_matrix, inv_matrix, q,
                                                   matrices, repetitions);
}
#endif

/*
  Returns a random floating-point value between min and max
*/
float RandomValueInInterval(float min, float max) {
  return min + static_cast<float>(rand()) /
                   (static_cast<float>(RAND_MAX) / (max - min));
}

/*
  returns if both the real and complex parts of the given ac_complex
  value are finite
*/
bool IsFinite(ac_complex<float> val) {
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  returns if the given value is finite
*/
bool IsFinite(float val) { return std::isfinite(val); }

/*
  Generate a random matrix M with a given epsilon such that
  cond(M, inf) <= (1+epsilon)/(1-epsilon)
  This is helpful as having a condition number with infinite norm close to 1
  reduces the numerical instability of the matrix inversion.
  Provided an epsilon value, this function populates the output vector with
  a matrix in a row fashion.

  Algorithm courtesy of Carl Christian Kjelgaard Mikkelsen (spock@cs.umu.se)

  Matlab code snipet this function reimplements in C++:

    function [A, B]=myDiagonal(m,epsilon)

    % Returns a matrix that is diagonally dominant by rows
    %
    % CALL SEQUENCE:
    %    [A, B]=ccDiagonally(m, epsilon)
    %
    % INPUT:
    %    m        the dimension
    %    epsilon  the dominance factor epsilon in (0,1]
    %
    % OUTPUT:
    %    A        a matrix which is strictly diagonally domimant by rows
    %    B        B = D\A, where D is the diagonal of A
    %
    % The main purpose of this function is to construct test matrices
    % for, say, Gaussian elimination with no pivoting.
    %
    % The matrix A is not necessarily well-conditioned, but
    % the infinity norm condition number of the matrix B is bounded by
    %
    %                  (1+epsilon)/(1 - epsilon)

    % PROGRAMMING by Carl Christian Kjelgaard Mikkelsen (spock@cs.umu.se)
    %   2021-10-21  Initial programming and testing.

    % Generate a random matrix
    R=rand(m,m)-rand(m,m);

    % Eliminate the diagonal
    D=diag(diag(R)); R=R-D;

    % Measure the weight of the off diagonal entries
    w=abs(R)*ones(m,1);

    % Construct the new diagonal elements
    d=w/epsilon;

    % Construct the matrix which is diagonally dominant
    A=R+diag(d);

    % Do the diagonal scaling
    B=diag(diag(A))\A;
*/
template <int size, typename T>
void GenerateMatrixWithCondititionNumber(float epsilon,
                                         std::vector<T> &output) {
  // Random min and max values for the random floating-point value generation
  constexpr float kRandomMin = 0;
  constexpr float kRandomMax = 1;

  // Generate a random matrix R with diagonal elements set to 0
  // and measure the weights of the off diagonal entries
  std::vector<T> r, weights;
  r.resize(size * size);
  weights.resize(size);
  for (int row = 0; row < size; row++) {
    weights[row] = {0};
    for (int col = 0; col < size; col++) {
      if (col != row) {
        float random1 = RandomValueInInterval(kRandomMin, kRandomMax);
        T elem;
#if COMPLEX == 1
        float random1I = RandomValueInInterval(kRandomMin, kRandomMax);
        elem = {random1, random1I};
        r[row * size + col] = elem;
#else
        elem = random1;
        r[row * size + col] = elem;
#endif
        weights[row] += elem;
      }
    }

    // Construct the new diagonal element
    weights[row] /= epsilon;
    r[row * size + row] = weights[row];
  }

  // Perform the diagonal scaling by solving:
  // diag(diag(A))*output = A
  for (int row = 0; row < size; row++) {
    for (int col = 0; col < size; col++) {
      output[row * size + col] = r[row * size + col] / r[row * size + row];
    }
  }
}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRows = ROWS_COMPONENT_V;
  constexpr size_t kColumns = COLS_COMPONENT_V;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kInverseMatrixSize = kRows * kColumns;
  constexpr bool kComplex = COMPLEX != 0;

#if defined(FPGA_SIMULATOR)
  constexpr size_t kMatricesToInvert = 1;
#else
  constexpr size_t kMatricesToInvert = 8;
#endif

  // Get the number of times we want to repeat the inversion
  // from the command line.
#if defined(FPGA_EMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 16;
#elif defined(FPGA_SIMULATOR)
  int repetitions = argc > 1 ? atoi(argv[1]) : 1;
#else
  int repetitions = argc > 1 ? atoi(argv[1]) : 6553600;
#endif
  if (repetitions < 1) {
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
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
    sycl::property_list
                    queue_properties{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(selector,
                                fpga_tools::exception_handler,
                                queue_properties);

    sycl::device device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    using TF = std::conditional_t<kComplex, ac_complex<float>, float>;
    // Select a type for computing the inverse in the testbench using a more
    // precise format than the kernel
    using TD = std::conditional_t<kComplex, ac_complex<double>, double>;

    // Create vectors to hold all the input and output matrices
    std::vector<TF> a;
    std::vector<TF> inv_matrix;
    std::vector<TF> precomputed_inv_matrix;

    a.resize(kMatricesToInvert * kAMatrixSize);
    inv_matrix.resize(kMatricesToInvert * kInverseMatrixSize);
    precomputed_inv_matrix.resize(kMatricesToInvert * kInverseMatrixSize);

    std::cout << "Generating " << kMatricesToInvert << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << ((kMatricesToInvert == 1) ? "x " : "ces ")
              << "of size "
              << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random input matrices and precompute their inverse
    srand(kRandomSeed);
    for (size_t i = 0; i < kMatricesToInvert; i++) {
      std::vector<TF> random_matrix;
      random_matrix.resize(kAMatrixSize);
      // Setting an epsilon of 0.5 ensures that the inverse matrix will have
      // a condition number using the infinite norm lower than 1.5/0.5 = 3
      float epsilon = 0.5;
      GenerateMatrixWithCondititionNumber<kRows>(epsilon, random_matrix);

      // Copy the generated matrix in the A vector
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          a[i * kAMatrixSize + col * kRows + row] =
              random_matrix[row * kColumns + col];
        }
      }

      // Precompute the inverse of A using the Gaussian elimination
      // A copy of A that will be modified
      TD a_copy[kColumns][kRows];
      // The inverse matrix that will be iteratively constructed starting from
      // the identity matrix
      TD inverse[kColumns][kRows];

      // Copy A in a_copy and set "inverse" to the identity matrix
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          if (row == col) {
            inverse[row][col] = {1.0};
          } else {
            inverse[row][col] = {0.0};
          }
          a_copy[row][col] = random_matrix[row * kColumns + col];
        }
      }

      // If we can't find a solution using the Gaussian elimination,
      // we may give up on this matrix and generate another one
      bool give_up = false;

      // Perform the Gaussian elimination
      for (int row = 0; row < kRows; row++) {
        // Find the next pivot
        auto pivot = a_copy[row][row];

        // If the pivot is zero, we need to swap the current row with
        // another row that would give a non-zero pivot.
        bool pivot_is_zero = pivot == 0.0 || pivot == -0.0;
        if (pivot_is_zero) {
          // Find an alternate row to use for pivoting
          for (int next_row = row + 1; next_row < kRows; next_row++) {
            TD potential_pivot = a_copy[next_row][row];
            bool potential_pivot_is_zero =
                potential_pivot == 0.0 || potential_pivot == -0.0;
            // row can be used to swap
            if (!potential_pivot_is_zero) {
              // Swap the two rows
              for (int j = 0; j < kColumns; j++) {
                auto tmp = a_copy[row][j];
                a_copy[row][j] = a_copy[next_row][j];
                a_copy[next_row][j] = tmp;

                tmp = inverse[row][j];
                inverse[row][j] = inverse[next_row][j];
                inverse[next_row][j] = tmp;
              }

              // The swap was successful, stop searching for a row to swap with
              break;
            }
          }

          // Get the new pivot
          pivot = a_copy[row][row];

          // If the swapping was unsuccessful are the new pivot is 0,
          // give up on this matrix generate another one
          give_up = pivot == 0.0 || pivot == -0.0;
          if (give_up) {
            break;
          }
        }

        // Divide the current row by the pivot value
        for (int k = 0; k < kColumns; k++) {
          a_copy[row][k] = a_copy[row][k] / pivot;
          inverse[row][k] = inverse[row][k] / pivot;
        }

        // Eliminate the current row in all other rows
        for (int row_to_eliminate = kRows - 1; row_to_eliminate >= 0;
             row_to_eliminate--) {
          if (row_to_eliminate == row) {
            continue;
          }

          auto factor = a_copy[row_to_eliminate][row];
          for (int k = 0; k < kColumns; k++) {
            if (k == row) {
              a_copy[row_to_eliminate][k] = 
                  a_copy[row_to_eliminate][k] - factor;
            } else {
              a_copy[row_to_eliminate][k] =
                  a_copy[row_to_eliminate][k] - (a_copy[row][k] * factor);
            }
            inverse[row_to_eliminate][k] =
                inverse[row_to_eliminate][k] - (inverse[row][k] * factor);
          }
        }
      }

      // Compute the norm inf of both the input and the inverse matrices
      // to compute the condition number and verify that it is lower than the
      // expected threshold
      double norm_inf_a = 0.0;
      double norm_inf_inverse = 0.0;
      for (size_t row = 0; row < kRows; row++) {
        // Compute the norm inf of the current row on both matrices
        double norm_current_row_of_a = 0.0;
        double norm_current_row_of_inverse = 0.0;
        for (size_t col = 0; col < kColumns; col++) {
          norm_current_row_of_a += abs(random_matrix[row * kColumns + col]);
          norm_current_row_of_inverse += abs(inverse[row][col]);
        }

        // Update the norm inf of both matrices if the norm inf of the current
        // row is the new max
        if (norm_current_row_of_a > norm_inf_a) {
          norm_inf_a = norm_current_row_of_a;
        }
        if (norm_current_row_of_inverse > norm_inf_inverse) {
          norm_inf_inverse = norm_current_row_of_inverse;
        }
      }

      // Copy the current inverse matrix in the precomputed_inv_matrix vector
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          // If any of the element in not finite, give up on this matrix
          if (!IsFinite(inverse[row][col])) {
            give_up = true;
          } else {
            precomputed_inv_matrix[i * kAMatrixSize + row * kColumns + col] =
                inverse[row][col];
          }
        }
      }

      // Compute the condition number
      double condition_number = norm_inf_a * norm_inf_inverse;
      double expected_conditionNumber = (1 + epsilon) / (1 - epsilon);

      // Regenerate this matrix if:
      // - the condition number is higher than the expected one
      // - we gave up earlier
      if (condition_number > expected_conditionNumber || give_up) {
        i--;
      }
#ifdef DEBUG
      else {
        std::cout << "A matrix" << std::endl;
        for (size_t row = 0; row < kRows; row++) {
          for (size_t col = 0; col < kColumns; col++) {
            std::cout << a[i * kAMatrixSize + col * kColumns + row] << " ";
          }
          std::cout << std::endl;
        }
        std::cout << "norm_inf_a " << norm_inf_a << std::endl;
        std::cout << "norm_inf_inverse " << norm_inf_inverse << std::endl;
        std::cout << "condition_number " << condition_number << std::endl;
      }
#endif
    }

    std::cout << "Running QR inversion of " << kMatricesToInvert << " matri"
              << ((kMatricesToInvert == 1) ? "x " : "ces ")
              << repetitions << " time"
              << ((repetitions > 1) ? "s" : "") 
              << std::endl;

    // Launch the compute kernel
    QRI(a, inv_matrix, q, kMatricesToInvert, repetitions);

    // Count the number of errors found for this matrix
    int error_count = 0;
    // Keep track of the max difference between the precomputed matrix using the
    // Gaussian elimination on the double datatype and the kernel computed
    // inverse matrix using a QR based algorithm with the float datatype.
    double max_diff_between_soft_and_hard = 0.0;

    // For output post-processing (OP)
    TF inv_matrix_op[kRows][kColumns];

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;

    std::cout << "Verifying results... ";
    for (int matrix = 0; matrix < kMatricesToInvert; matrix++) {

      // Read the inverse matrix from the output vector to inv_matrix_op
      size_t idx = 0;
      for (size_t j = 0; j < kColumns; j++) {
        for (size_t i = 0; i < kRows; i++) {
          inv_matrix_op[j][i] = inv_matrix[matrix * kInverseMatrixSize + idx];
          idx++;
        }
      }

#ifdef DEBUG
      std::cout << "Kernel inverse" << std::endl;
      for (int row = 0; row < kRows; row++) {
        for (int col = 0; col < kColumns; col++) {
          std::cout << inv_matrix_op[row][col] << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "Precomputed inverse" << std::endl;
      for (int row = 0; row < kRows; row++) {
        for (int col = 0; col < kColumns; col++) {
          std::cout << precomputed_inv_matrix[matrix * kAMatrixSize +
                                            row * kColumns + col]
                    << " ";
        }
        std::cout << std::endl;
      }
#endif

      // Keep track of the max difference between the precomputed inverse and
      // the kernel inverse
      double max_diff = 0.0;

#if COMPLEX == 1
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          double diff_r = abs(
              inv_matrix_op[row][col].r() -
      precomputed_inv_matrix[matrix * kAMatrixSize + row * kColumns + col].r());

          double diff_i = abs(
              inv_matrix_op[row][col].i() -
      precomputed_inv_matrix[matrix * kAMatrixSize + row * kColumns + col].i());

          if (!std::isfinite(diff_r) || !std::isfinite(diff_r)) {
            error_count++;
          }

          if (diff_r > max_diff) {
            max_diff = diff_r;
          }
          if (diff_i > max_diff) {
            max_diff = diff_i;
          }

          if (diff_r > kErrorThreshold) {
            error_count++;
          }
          if (diff_i > kErrorThreshold) {
            error_count++;
          }
        }
      }
#else
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          double diff = abs(
              inv_matrix_op[i][j] -
              precomputed_inv_matrix[matrix * kAMatrixSize + i * kColumns + j]);

          if (!std::isfinite(diff)) {
            error_count++;
          }

          if (diff > max_diff) {
            max_diff = diff;
          }

          if (diff > kErrorThreshold) {
            error_count++;
          }
        }
      }
#endif

      // Update the max diff
      if (max_diff > max_diff_between_soft_and_hard) {
        max_diff_between_soft_and_hard = max_diff;
      }

      // If an error was found, stop checking matrices
      if (error_count > 0) {
        break;
      }
    }  // end of matrix

    if (error_count > 0) {
      std::cout << std::endl << "FAILED" << std::endl;
      std::cout << std::endl
                << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
      std::cout << "Max difference between the precomputed inverse and the "
                << "kernel value: " << max_diff_between_soft_and_hard 
                << std::endl;
      return 1;
    }

    std::cout << std::endl << "PASSED" << std::endl;
    return 0;

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
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: "
              << e.what() << std::endl;
    std::cerr << "   You can reduce the memory requirement by reducing the "
                 "number of matrices generated. Specify a smaller number when "
                 "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << (((long long)kMatricesToInvert 
                * (kAMatrixSize + kInverseMatrixSize) 
                * sizeof(float)) / pow(2, 30))
              << " GBs of memory was requested for " << kMatricesToInvert
              << " matrices, each of size " << kRows << " x " << kColumns
              << std::endl;
    std::terminate();
  }
}
