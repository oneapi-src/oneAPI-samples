#include <math.h>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <chrono>
#include <list>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#include "qrd.hpp"

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT and FIXED_ITERATIONS are defined
  by the build system.
  Depending on the value of COMPLEX, the real or complex QRDecomposition

  Function arguments:
  - AMatrix:    The input matrix. Interpreted as a transposed matrix.
  - QMatrix:    The Q matrix. The function will overwrite this matrix.
  - RMatrix     The R matrix. The function will overwrite this matrix.
                The vector will only contain the upper triangular elements
                of the matrix, in a row by row fashion.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the AMatrix 
                vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)
*/

#if COMPLEX == 0
// Real single precision floating-point QR Decomposition
void QRDecomposition( std::vector<float> &AMatrix, 
                      std::vector<float> &QMatrix,
                      std::vector<float> &RMatrix,
                      sycl::queue &q, 
                      size_t matrices, 
                      size_t reps) {
  constexpr bool isComplex = false;
  QRDecomposition_impl< COLS_COMPONENT, 
                        ROWS_COMPONENT, 
                        FIXED_ITERATIONS, 
                        isComplex, 
                        float>(AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
#else
// Complex single precision floating-point QR Decomposition
void QRDecomposition( std::vector<ac_complex<float>> &AMatrix, 
                      std::vector<ac_complex<float>> &QMatrix,
                      std::vector<ac_complex<float>> &RMatrix,
                      sycl::queue &q, 
                      size_t matrices, 
                      size_t reps) {
  constexpr bool isComplex = true;
  QRDecomposition_impl< COLS_COMPONENT, 
                        ROWS_COMPONENT, 
                        FIXED_ITERATIONS, 
                        isComplex, 
                        float>(AMatrix, QMatrix, RMatrix, q, matrices, reps); 
}
#endif

/*
  returns if both the real and complex parts of the given ac_complex
  value are finite
*/
bool isFinite(ac_complex<float> val){
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

/*
  returns if the given value is finite
*/
bool isFinite(float val){
  return std::isfinite(val);
}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kQMatrixSize = kRows * kColumns;
  constexpr size_t kRMatrixSize = kColumns * (kColumns + 1) / 2;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;
  constexpr bool kComplex = COMPLEX != 0;

  // Get the number of random matrices to decompose from the command line
  // If no value is given, will only decompose 1 random matrix
  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    std::cout << "Must run at least 1 matrix"  << std::endl;
    return 1;
  }

  try {
    // SYCL boilerplate
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif
    sycl::queue q = sycl::queue(device_selector, dpc_common::exception_handler);
    sycl::device device = q.get_device();
    std::cout << "Device name: " 
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    // Select a type for this compile depending on the value of COMPLEX
    typedef typename std::conditional<kComplex, ac_complex<float>, 
                                                float>::type T;

    // Create vectors to hold all the input and ouput matrices
    std::vector<T> AMatrix;
    std::vector<T> QMatrix; 
    std::vector<T> RMatrix; 

    AMatrix.resize(matrices * kAMatrixSize);
    QMatrix.resize(matrices * kQMatrixSize);
    RMatrix.resize(matrices * kRMatrixSize);

    std::cout << "Generating " << matrices << " random ";
    if constexpr(kComplex){
      std::cout << "complex ";
    }
    else{
      std::cout << "real ";
    }
    std::cout << "matri"<< ((matrices == 1) ? "x " : "ces ") 
              << "of size " << kRows << "x" << kColumns << " "
              << std::endl;

    // Generate the random input matrices
    srand(kRandomSeed);
    for (size_t i = 0; i < matrices; i++) {
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          float random_real = rand() % (kRandomMax - kRandomMin) + kRandomMin;
#if COMPLEX == 0
            AMatrix[i * kAMatrixSize + col * kRows + row] = random_real;
#else
            float random_imag = rand() % (kRandomMax - kRandomMin) + kRandomMin;          
            ac_complex<float> random_complex{random_real, random_imag};
            AMatrix[i * kAMatrixSize + col * kRows + row] = random_complex;
#endif
        } // end of col
      } // end of row
    } // end of i

#ifdef DEBUG
    for (size_t i = 0; i < matrices; i++) {
      std::cout << "A MATRIX " << i << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          std::cout << AMatrix[i * kAMatrixSize + col * kRows + row] << " "
        } // end of col
        std::cout << std::endl;
      } // end of row      
    } // end of i
#endif

#if defined(FPGA_EMULATOR)
    size_t reps = 1;
#else
    // Accelerator warmup
    QRDecomposition(AMatrix, QMatrix, RMatrix, q, 2048, 1); 
    size_t reps = 32;
#endif
    std::cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << std::endl;

    // Launch the compute kernel and time the execution
    std::chrono::high_resolution_clock::time_point start_time = 
                                      std::chrono::high_resolution_clock::now();
    QRDecomposition(AMatrix, QMatrix, RMatrix, q, matrices, reps);
    std::chrono::high_resolution_clock::time_point end_time = 
                                      std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    q.throw_asynchronous();

    std::cout << "   Total duration:   " << diff.count() << " s" << std::endl;
    std::cout << "Throughput: " << reps * matrices / diff.count() / 1000
              << "k matrices/s" << std::endl;

    // Indexes of matrices to check
    std::list<size_t> to_check;
    // We will check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (matrices > 2) to_check.push_back(matrices / 2);
    if (matrices > 1) to_check.push_back(matrices - 1);

    // For output post-processing (OP)
    T QMatrixOP[kRows][kColumns];
    T RMatrixOP[kColumns][kColumns];

    // For rectangular matrices, Q is only going to have orthogonal columns 
    // so we won't check if the rows are orthogonal
    bool squareMatrices = kRows == kColumns;

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;
    // The orthogonality check is more sensible to numerical error, the 
    // threshold is then set a bit higher
    float QOrthoErrorThreshold = pow(2.0, -9);

    std::cout << "Verifying results on matrix ";
    // Go over the matrices to check
    for (size_t matrix : to_check) {
      std::cout << matrix << std::endl;

      // keep track of Q and R element indexes 
      size_t RIdx = 0;
      size_t QIdx = 0;

      // Read the R matrix from the output vector to the RMatrixOP matrix
      for (size_t i = 0; i < kColumns; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          if (j < i)
            RMatrixOP[i][j] = 0;
          else {
            RMatrixOP[i][j] = RMatrix[matrix * kRMatrixSize + RIdx];
            RIdx++;
          }
        }
      }

      // Read the Q matrix from the output vector to the QMatrixOP matrix
      for (size_t j = 0; j < kColumns; j++) {
        for (size_t i = 0; i < kRows; i++) {
          QMatrixOP[i][j] = QMatrix[matrix * kQMatrixSize + QIdx];
          QIdx++;
        }
      }

#ifdef DEBUG
      std::cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << RMatrixOP[i][j] << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "Q MATRIX" << std::endl;
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          std::cout << QMatrixOP[i][j] << " ";
        }
        std::cout << std::endl;
      }
#endif

      // Count the number of errors found for this matrix
      size_t count = 0;
      bool error = false;

      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          // Compute Q * R at index i,j
          T QRij{0};
          for (size_t k = 0; k < kColumns; k++) {
            QRij += QMatrixOP[i][k] * RMatrixOP[k][j];
          }

          // Compute transpose(Q) * Q at index i,j
          T QtQij{0};
          if(i<kColumns){
            for (size_t k = 0; k < kRows; k++) {
#if COMPLEX == 0
              QtQij += QMatrixOP[k][i] * QMatrixOP[k][j];
#else
              QtQij += QMatrixOP[k][i] * QMatrixOP[k][j].conj();
#endif
            }
          }

          // Compute Q * transpose(Q) at index i,j
          T QQtij{0};
          if(squareMatrices){
            if(i<kColumns){
              for (size_t k = 0; k < kRows; k++) {
#if COMPLEX == 0
                QQtij += QMatrixOP[i][k] * QMatrixOP[j][k];
#else
                QQtij += QMatrixOP[i][k] * QMatrixOP[j][k].conj();
#endif
              }
            }
          }

          // Verify that all the results are OK:
          // Q * R = A at index i,j
          bool QREqA;
          // transpose(Q) * Q = Id at index i,j
          bool QtQEqId;
          // Q * transpose(Q) = Id at index i,j
          bool QQtEqId;
          // R is upped triangular
          bool RIsUpperTriang;
          // R is finite at index i,j
          bool RIsFinite;

#if COMPLEX == 0
          QREqA = abs(AMatrix[matrix * kAMatrixSize + j * kRows + i] - QRij) 
                  < kErrorThreshold;

          QtQEqId = ((i == j) && (abs(QtQij - 1) < QOrthoErrorThreshold)) ||
                    ((i != j) && (abs(QtQij) < QOrthoErrorThreshold));

          QQtEqId = !squareMatrices || 
                    (((i == j) && (abs(QQtij - 1) < QOrthoErrorThreshold)) ||
                    ((i != j) && (abs(QQtij) < QOrthoErrorThreshold)));

          RIsUpperTriang =  (i >= kColumns) || 
                    ((i > j) && ((abs(RMatrixOP[i][j]) < kErrorThreshold) )) ||
                            ((i <= j));

          RIsFinite = (i < kColumns) && isFinite(RMatrixOP[i][j]);
#else
          QREqA = (abs(AMatrix[matrix * kAMatrixSize + j * kRows + i].r()
                                              - QRij.r()) < kErrorThreshold)
                  && (abs(AMatrix[matrix * kAMatrixSize + j * kRows + i].i() 
                                              - QRij.i()) < kErrorThreshold);

          QtQEqId = (
                    ((i == j) && (abs(QtQij.r() - 1) < QOrthoErrorThreshold)) ||
          (((i != j) || (j>=kRows)) && (abs(QtQij.r()) < QOrthoErrorThreshold))
                    ) && (abs(QtQij.i()) < QOrthoErrorThreshold);

          QQtEqId = !squareMatrices || 
                    (
                      (
                    ((i == j) && (abs(QQtij.r() - 1) < QOrthoErrorThreshold)) || 
          (((i != j) || (j>=kRows)) && (abs(QQtij.r()) < QOrthoErrorThreshold))
                      )
                      && (abs(QQtij.i()) < QOrthoErrorThreshold)
                    );

          RIsUpperTriang =  (i >= kColumns) || 
                            (
                              (i > j) && 
                              (
                                (abs(RMatrixOP[i][j].r()) < kErrorThreshold) &&
                                (abs(RMatrixOP[i][j].i()) < kErrorThreshold)
                              )
                            )
                            || (i <= j);

          RIsFinite = (i < kColumns) && isFinite(RMatrixOP[i][j]);
#endif

          // If any of the checks failed
          if (!QREqA ||
              !QtQEqId ||
              !QQtEqId ||
              !RIsUpperTriang||
              !isFinite(QRij)||
              !isFinite(QtQij)||
              !isFinite(QQtij)||
              !RIsFinite
            ) {

            // Increase the error count for this matrix
            count++;

            // Continue counting the errors even if we now we are going to 
            // produce an error
            if(error){
              continue;
            }

            if(!QREqA){
              std::cout << "Error: A[" << i << "][" << j << "] = " << 
                        AMatrix[matrix * kAMatrixSize + j * kRows + i]
                        << " but QR[" << i << "][" << j << "] = " << QRij 
                        << std::endl;
            }
            if(!QREqA) {
              std::cout << "The difference is greater than tolerated (" 
                        << kErrorThreshold << ")" << std::endl;
            }
            if(!QtQEqId || !QQtEqId) {
              std::cout << "Q is not orthogonal at i " << i << " j " << j 
                        << ":" << std::endl  
                        << " transpose(Q) * Q = " << QtQij << std::endl 
                        << " Q * transpose(Q) =" << QQtij << std::endl;
              std::cout << "kQOrthoErrorThreshold = " << QOrthoErrorThreshold 
                        << std::endl;       
            }
            if(!RIsUpperTriang) {
              std::cout << "R is not upper triangular at i " << i << " j " << j
                        << ":" << std::endl
                        << " R = " << RMatrixOP[i][j] << std::endl;
            }
            if(!isFinite(QRij)) {
              std::cout << "QR[" << i << "][" << j << "] = " << QRij 
                        << " is not finite" << std::endl;
            }
            if(!isFinite(QtQij)) {
              std::cout << "transpose(Q) * Q at i " << i << " j " << j 
                        << " = " << QtQij << " is not finite" << std::endl;
            }
            if(!isFinite(QQtij)) {
              std::cout << "Q * transpose(Q) at i " << i << " j " << j 
                        << " = " << QQtij << " is not finite" << std::endl;
            }
            if(!RIsFinite) {
              std::cout << "R[" << i << "][" << j << "] = " << RMatrixOP[i][j] 
                        << " is not finite" << std::endl;
            }
            error = true;
          }
        } // end of j
      } // end of i

      if (count > 0) {
        std::cout << std::endl << "FAILED" << std::endl;
        std::cout << std::endl << "!!!!!!!!!!!!!! " << count << " errors" 
                  << std::endl;
        return 1;
      }
    }

    std::cout << std::endl << "PASSED" << std::endl;
    return 0;

  } catch (sycl::exception const &e) {
    std::cerr << "Caught a synchronous SYCL exception: " << e.what() 
              << std::endl;
    std::cerr <<  "   If you are targeting an FPGA hardware, "
                  "ensure that your system is plugged to an FPGA board that is "
                  "set up correctly"
              << std::endl;
    std::cerr <<  "   If you are targeting the FPGA emulator, compile with "
                  "-DFPGA_EMULATOR"
              << std::endl;

    std::terminate();
  } catch (std::bad_alloc const &e) {
    std::cerr << "Caught a memory allocation exception on the host: " 
              << e.what() << std::endl;
    std::cerr <<  "   You can reduce the memory requirement by reducing the "
                  "number of matrices generated. Specify a smaller number when "
                  "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << ((long long)matrices * (kAMatrixSize + kQRMatrixSize) *
                  sizeof(float)) / pow(2, 30)
              << " GBs of memory was requested for " << matrices
              << " matrices, each of size " << kRows << " x " << kColumns 
              << std::endl;
    std::terminate();
  }
} // end of main
