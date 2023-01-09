#include <math.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include <list>

#include "exception_handler.hpp"

#include "qrd.hpp"

#ifdef FPGA_SIMULATOR
#define ROWS_COMPONENT_V 8
#define COLS_COMPONENT_V 8
#else
#define ROWS_COMPONENT_V ROWS_COMPONENT
#define COLS_COMPONENT_V COLS_COMPONENT
#endif

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex QRDecomposition is
  defined

  Function arguments:
  - a_matrix:    The input matrix. Interpreted as a transposed matrix.
  - q_matrix:    The Q matrix. The function will overwrite this matrix.
  - r_matrix     The R matrix. The function will overwrite this matrix.
                 The vector will only contain the upper triangular elements
                 of the matrix, in a row by row fashion.
  - q:           The device queue.
  - matrix_count: Number of matrices to decompose.
  - repetitions: The number of repetitions of the computation to execute.
                 (for performance evaluation)
*/
#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void QRDecomposition(std::vector<float> &a_matrix, std::vector<float> &q_matrix,
                     std::vector<float> &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = false;
  QRDecompositionImpl<COLS_COMPONENT_V, ROWS_COMPONENT_V, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);
}
#else
// Complex single precision floating-point QR Decomposition
void QRDecomposition(std::vector<ac_complex<float> > &a_matrix,
                     std::vector<ac_complex<float> > &q_matrix,
                     std::vector<ac_complex<float> > &r_matrix, sycl::queue &q,
                     int matrix_count,
                     int repetitions) {
  constexpr bool is_complex = true;
  QRDecompositionImpl<COLS_COMPONENT_V, ROWS_COMPONENT_V, FIXED_ITERATIONS,
                       is_complex, float>(a_matrix, q_matrix, r_matrix, q,
                                          matrix_count, repetitions);
}
#endif

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

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kRows = ROWS_COMPONENT_V;
  constexpr size_t kColumns = COLS_COMPONENT_V;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kQMatrixSize = kRows * kColumns;
  constexpr size_t kRMatrixSize = kColumns * (kColumns + 1) / 2;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;
  constexpr bool kComplex = COMPLEX != 0;

#if defined(FPGA_SIMULATOR)
  std::cout << "Using 32x32 matrices for simulation to reduce runtime" << std::endl;
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
    std::cout << "Number of repetitions given is lower that 1." << std::endl;
    std::cout << "The decomposition must occur at least 1 time." << std::endl;
    std::cout << "Increase the number of repetitions (e.g. 16)." << std::endl;
    return 1;
  }

#if defined(FPGA_SIMULATOR)
  constexpr size_t kMatricesToDecompose = 1;
#else
  constexpr size_t kMatricesToDecompose = 8;
#endif

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
    using T = std::conditional_t<kComplex, ac_complex<float>, float>;

    // Create vectors to hold all the input and output matrices
    std::vector<T> a_matrix;
    std::vector<T> q_matrix;
    std::vector<T> r_matrix;

    a_matrix.resize(kAMatrixSize * kMatricesToDecompose);
    q_matrix.resize(kQMatrixSize * kMatricesToDecompose);
    r_matrix.resize(kRMatrixSize * kMatricesToDecompose);

    std::cout << "Generating " << kMatricesToDecompose << " random ";
    if constexpr (kComplex) {
      std::cout << "complex ";
    } else {
      std::cout << "real ";
    }
    std::cout << "matri" << (kMatricesToDecompose > 1 ? "ces" : "x")
              << " of size "
              << kRows << "x" << kColumns << " " << std::endl;

    // Generate the random input matrices
    srand(kRandomSeed);

    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          float random_real = rand() % (kRandomMax - kRandomMin) + kRandomMin;
  #if COMPLEX == 0
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_real;
  #else
          float random_imag = rand() % (kRandomMax - kRandomMin) + kRandomMin;
          ac_complex<float> random_complex{random_real, random_imag};
          a_matrix[matrix_index * kAMatrixSize
                 + col * kRows + row] = random_complex;
  #endif
        }  // end of col
      }    // end of row

  #ifdef DEBUG
      std::cout << "A MATRIX " << matrix_index << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << a_matrix[matrix_index * kAMatrixSize
                              + col * kRows + row] << " ";
        }  // end of col
        std::cout << std::endl;
      }  // end of row
  #endif

    } // end of matrix_index

    std::cout << "Running QR decomposition of " << kMatricesToDecompose
              << " matri" << (kMatricesToDecompose > 1 ? "ces " : "x ")
              << repetitions << " times" << std::endl;

    QRDecomposition(a_matrix, q_matrix, r_matrix, q, kMatricesToDecompose,
                                                                  repetitions);

    // For output post-processing (op)
    T q_matrix_op[kRows][kColumns];
    T r_matrix_op[kRows][kColumns];

    // For rectangular matrices, Q is only going to have orthogonal columns
    // so we won't check if the rows are orthogonal
    bool square_matrices = kRows == kColumns;

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;
    // The orthogonality check is more sensible to numerical error, the
    // threshold is then set a bit higher
    float q_ortho_error_threshold = pow(2.0, -9);

    // Check Q and R matrices
    std::cout << "Verifying results...";
    for(int matrix_index = 0; matrix_index < kMatricesToDecompose;
                                                                matrix_index++){

      // keep track of Q and R element indexes
      size_t r_idx = 0;
      size_t q_idx = 0;

      // Read the R matrix from the output vector to the RMatrixOP matrix
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          if (j < i)
            r_matrix_op[i][j] = 0;
          else {
            r_matrix_op[i][j] = r_matrix[matrix_index*kRMatrixSize
                                       + r_idx];
            r_idx++;
          }
        }
      }

      // Read the Q matrix from the output vector to the QMatrixOP matrix
      for (size_t j = 0; j < kColumns; j++) {
        for (size_t i = 0; i < kRows; i++) {
          q_matrix_op[i][j] = q_matrix[matrix_index*kQMatrixSize
                                     + q_idx];
          q_idx++;
        }
      }

  #ifdef DEBUG
      std::cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << r_matrix_op[i][j] << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "Q MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << q_matrix_op[i][j] << " ";
        }
        std::cout << std::endl;
      }
  #endif

      // Count the number of errors found for this matrix
      size_t error_count = 0;
      bool error = false;

      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          // Compute Q * R at index i,j
          T q_r_ij{0};
          for (size_t k = 0; k < kColumns; k++) {
            q_r_ij += q_matrix_op[i][k] * r_matrix_op[k][j];
          }

          // Compute transpose(Q) * Q at index i,j
          T qt_q_ij{0};
          if (i < kColumns) {
            for (size_t k = 0; k < kRows; k++) {
  #if COMPLEX == 0
              qt_q_ij += q_matrix_op[k][i] * q_matrix_op[k][j];
  #else
              qt_q_ij += q_matrix_op[k][i] * q_matrix_op[k][j].conj();
  #endif
            }
          }

          // Compute Q * transpose(Q) at index i,j
          T q_qt_ij{0};
          if (square_matrices) {
            if (i < kColumns) {
              for (size_t k = 0; k < kRows; k++) {
  #if COMPLEX == 0
                q_qt_ij += q_matrix_op[i][k] * q_matrix_op[j][k];
  #else
                q_qt_ij += q_matrix_op[i][k] * q_matrix_op[j][k].conj();
  #endif
              }
            }
          }

          // Verify that all the results are OK:
          // Q * R = A at index i,j
          bool q_r_eq_a;
          // transpose(Q) * Q = Id at index i,j
          bool qt_q_eq_id;
          // Q * transpose(Q) = Id at index i,j
          bool q_qt_eq_id;
          // R is upped triangular
          bool r_is_upper_triang;
          // R is finite at index i,j
          bool r_is_finite;

  #if COMPLEX == 0
          q_r_eq_a = abs(a_matrix[matrix_index * kAMatrixSize
                                + j * kRows + i]
                       - q_r_ij) < kErrorThreshold;

          qt_q_eq_id =
                  ((i == j) && (abs(qt_q_ij - 1) < q_ortho_error_threshold)) ||
                  ((i != j) && (abs(qt_q_ij) < q_ortho_error_threshold));

          q_qt_eq_id = !square_matrices ||
                  (((i == j) && (abs(q_qt_ij - 1) < q_ortho_error_threshold)) ||
                  ((i != j) && (abs(q_qt_ij) < q_ortho_error_threshold)));

          r_is_upper_triang =
              (i >= kColumns) ||
              ((i > j) && ((abs(r_matrix_op[i][j]) < kErrorThreshold))) ||
              ((i <= j));

  #else
          q_r_eq_a = (abs(a_matrix[matrix_index * kAMatrixSize
                                 + j * kRows + i].r() -
                       q_r_ij.r()) < kErrorThreshold) &&
                  (abs(a_matrix[matrix_index * kAMatrixSize
                              + j * kRows + i].i() -
                       q_r_ij.i()) < kErrorThreshold);

          qt_q_eq_id =
              (((i == j) && (abs(qt_q_ij.r() - 1) < q_ortho_error_threshold)) ||
(((i != j) || (j >= kRows)) && (abs(qt_q_ij.r()) < q_ortho_error_threshold))) &&
              (abs(qt_q_ij.i()) < q_ortho_error_threshold);

          q_qt_eq_id =
              !square_matrices ||
            ((((i == j) && (abs(q_qt_ij.r() - 1) < q_ortho_error_threshold)) ||
                (((i != j) || (j >= kRows)) &&
                 (abs(q_qt_ij.r()) < q_ortho_error_threshold))) &&
               (abs(q_qt_ij.i()) < q_ortho_error_threshold));

          r_is_upper_triang =
              (i >= kColumns) ||
              ((i > j) && ((abs(r_matrix_op[i][j].r()) < kErrorThreshold) &&
                           (abs(r_matrix_op[i][j].i()) < kErrorThreshold))) ||
              (i <= j);

  #endif

          r_is_finite =
            ((i < kColumns) && IsFinite(r_matrix_op[i][j])) || (i >= kColumns);

          // If any of the checks failed
          if (!q_r_eq_a || !qt_q_eq_id || !q_qt_eq_id || !r_is_upper_triang ||
              !IsFinite(q_r_ij) || !IsFinite(qt_q_ij) || !IsFinite(q_qt_ij) ||
              !r_is_finite) {
            // Increase the error count for this matrix
            error_count++;

            // Continue counting the errors even if we now we are going to
            // produce an error
            if (error) {
              continue;
            }

            if (!q_r_eq_a) {
              std::cout << "Error: A[" << i << "][" << j << "] = "
                        << a_matrix[matrix_index * kAMatrixSize
                                  + j * kRows + i]
                        << " but QR[" << i << "][" << j << "] = " << q_r_ij
                        << std::endl;
            }
            if (!q_r_eq_a) {
              std::cout << "The difference is greater than tolerated ("
                        << kErrorThreshold << ")" << std::endl;
            }
            if (!qt_q_eq_id || !q_qt_eq_id) {
              std::cout << "Q is not orthogonal at i " << i << " j " << j << ":"
                        << std::endl
                        << " transpose(Q) * Q = " << qt_q_ij << std::endl
                        << " Q * transpose(Q) =" << q_qt_ij << std::endl;
              std::cout << "q_ortho_error_threshold = "
                        << q_ortho_error_threshold
                        << std::endl;
            }
            if (!r_is_upper_triang) {
              std::cout << "R is not upper triangular at i " << i << " j " << j
                        << ":" << std::endl
                        << " R = " << r_matrix_op[i][j] << std::endl;
            }
            if (!IsFinite(q_r_ij)) {
              std::cout << "QR[" << i << "][" << j << "] = " << q_r_ij
                        << " is not finite" << std::endl;
            }
            if (!IsFinite(qt_q_ij)) {
              std::cout << "transpose(Q) * Q at i " << i << " j " << j << " = "
                        << qt_q_ij << " is not finite" << std::endl;
            }
            if (!IsFinite(q_qt_ij)) {
              std::cout << "Q * transpose(Q) at i " << i << " j " << j << " = "
                        << q_qt_ij << " is not finite" << std::endl;
            }
            if (!r_is_finite) {
              std::cout << "R[" << i << "][" << j << "] = " << r_matrix_op[i][j]
                        << " is not finite" << std::endl;
            }
            error = true;
          }
        }  // end of j
      }    // end of i

      if (error_count > 0) {
        std::cout << std::endl << "FAILED" << std::endl;
        std::cout << std::endl
                  << "!!!!!!!!!!!!!! " << error_count << " errors" << std::endl;
        return 1;
      }
    } // end of matrix_index


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
              << ((kAMatrixSize + kQRMatrixSize) * 2 * kMatricesToDecompose
                 * sizeof(float)) / pow(2, 30)
              << " GBs of memory was requested for the decomposition of a "
              << "matrix of size " << kRows << " x " << kColumns
              << std::endl;
    std::terminate();
  }
}  // end of main
