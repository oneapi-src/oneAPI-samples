// ==============================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#include <math.h>

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <chrono>
#include <list>

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

#if COMPLEX == 0
void FloatQRDecomposition(  std::vector<float> &AMatrix, 
                            std::vector<float> &QMatrix,
                            std::vector<float> &RMatrix,
                            sycl::queue &q, size_t matrices, size_t reps);
#else
void ComplexFloatQRDecomposition( std::vector<ac_complex<float>> &AMatrix, 
                                  std::vector<ac_complex<float>> &QMatrix,
                                  std::vector<ac_complex<float>> &RMatrix,
                                  sycl::queue &q, size_t matrices, size_t reps);
#endif


bool isFinite(ac_complex<float> val){
  return std::isfinite(val.r()) && std::isfinite(val.i());
}

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

  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    std::cout << "Must run at least 1 matrix"  << std::endl;
    return 1;
  }

  try {
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

#if COMPLEX == 0
  std::cout << "Type is not complex" << std::endl;
#else
  std::cout << "Type is complex" << std::endl;
#endif

#if COMPLEX == 0
    std::vector<float> AMatrix;
    std::vector<float> QMatrix; 
    std::vector<float> RMatrix; 
#else
    std::vector<ac_complex<float>> AMatrix;
    std::vector<ac_complex<float>> QMatrix;
    std::vector<ac_complex<float>> RMatrix;
#endif

    AMatrix.resize(matrices * kAMatrixSize);
    QMatrix.resize(matrices * kQMatrixSize);
    RMatrix.resize(matrices * kRMatrixSize);

    std::cout << "Generating " << matrices << " random "
#if COMPLEX == 0
              << "real "
#else
              << "complex "
#endif
              << "matri"<< ((matrices == 1) ? "x " : "ces ") 
              << "of size " << kRows << "x" << kColumns << " "
              << std::endl;

    srand(kRandomSeed);

    for (size_t i = 0; i < matrices; i++) {
      // std::cout << "A MATRIX" << std::endl;
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          int val = rand();
          float random_real = val % (kRandomMax - kRandomMin) + kRandomMin;

#if COMPLEX == 0
          AMatrix[i * kAMatrixSize + col * kRows + row] = random_real;
#else
          val = rand();
          float random_imag = val % (kRandomMax - kRandomMin) + kRandomMin;          
          ac_complex<float> random_complex = {random_real, random_imag};
          AMatrix[i * kAMatrixSize + col * kRows + row] = random_complex;
#endif
          // std::cout << AMatrix[i * kAMatrixSize + col * kRows + row] 
          //      << " ";
        }
        // std::cout << std::endl;
      }
    }

    // }

#if defined(FPGA_EMULATOR)
#else
    // Accelerator warmup
#if COMPLEX == 0
    FloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, 1, 1); 
#else
    ComplexFloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, 1, 1); 
#endif
#endif

#if defined(FPGA_EMULATOR)
    size_t reps = 1;
#else
    size_t reps = 32;
#endif
    std::cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << std::endl;

    std::chrono::high_resolution_clock::time_point start_time = 
                                      std::chrono::high_resolution_clock::now();
#if COMPLEX == 0
    FloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, matrices, reps);
#else
    ComplexFloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, matrices, reps);
#endif
    std::chrono::high_resolution_clock::time_point end_time = 
                                      std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    q.throw_asynchronous();

    std::cout << "   Total duration:   " << diff.count() << " s"
         << std::endl;
    std::cout << "Throughput: " << reps * matrices / diff.count() / 1000
         << "k matrices/s"
         << std::endl;

    std::list<size_t> to_check;
    // We will check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (matrices > 2) to_check.push_back(matrices / 2);
    if (matrices > 1) to_check.push_back(matrices - 1);

    std::cout << "Verifying results on matrix";


    // For output-postprocessing (OP)
#if COMPLEX == 0
    float QMatrixOP[kRows][kColumns];
    float RMatrixOP[kColumns][kColumns];
#else
    ac_complex<float> QMatrixOP[kRows][kColumns];
    ac_complex<float> RMatrixOP[kColumns][kColumns];
#endif

    bool squareMatrices = kRows == kColumns;
    constexpr float kErrorThreshold = 1e-4;
    float QOrthoErrorThreshold = pow(2.0, -9);

    for (size_t matrix : to_check) {
      std::cout << " " << matrix << std::endl;
      size_t RIdx = 0;
      size_t QIdx = 0;

      // std::cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < kColumns; i++) {
        for (size_t j = 0; j < kColumns; j++) {
          if (j < i)
            RMatrixOP[i][j] = 0;
          else {
            RMatrixOP[i][j] = RMatrix[matrix * kRMatrixSize + RIdx];
            RIdx++;
          }
          // std::cout << RMatrixOP[i][j] << " ";
        }
        // std::cout << std::endl;
      }
      // std::cout << std::endl;

      for (size_t j = 0; j < kColumns; j++) {
        for (size_t i = 0; i < kRows; i++) {
          QMatrixOP[i][j] = QMatrix[matrix * kQMatrixSize + QIdx];
          QIdx++;
        }
      }

      // std::cout << "Q MATRIX" << std::endl;
      // for (size_t i = 0; i < kRows; i++) {
      //   for (size_t j = 0; j < kColumns; j++) {
      //     std::cout << QMatrixOP[i][j] << " ";
      //   }
      //   std::cout << std::endl;
      // }

      size_t count = 0;
      bool error = false;


      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {

#if COMPLEX == 0
          // Q * R
          float QRij = 0;
          // transpose(Q) * Q
          float QtQij = 0;
          // Q * transpose(Q)
          float QQtij = 0;
#else
          // Q * R
          ac_complex<float> QRij{0};
          // transpose(Q) * Q
          ac_complex<float> QtQij{0};
          // Q * transpose(Q)
          ac_complex<float> QQtij{0};
#endif

          for (size_t k = 0; k < kColumns; k++) {
            QRij += QMatrixOP[i][k] * RMatrixOP[k][j];
          }

          if(i<kColumns){
            for (size_t k = 0; k < kRows; k++) {
#if COMPLEX == 0
              QtQij += QMatrixOP[k][i] * QMatrixOP[k][j];
#else
              QtQij += QMatrixOP[k][i] * QMatrixOP[k][j].conj();
#endif
            }
          }

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

#if COMPLEX == 0
          bool QREqA = (abs(AMatrix[matrix * kAMatrixSize +
                              j * kRows + i] - QRij) 
                          < kErrorThreshold);

          bool QtQEqId = 
                      ((i == j) && (abs(QtQij - 1) < QOrthoErrorThreshold)) ||
                           ((i != j) && (abs(QtQij) < QOrthoErrorThreshold));

          bool QQtEqId =  !squareMatrices || 
                      (((i == j) && (abs(QQtij - 1) < QOrthoErrorThreshold)) ||
                          ((i != j) && (abs(QQtij) < QOrthoErrorThreshold)));

          bool RIsUpperTriang = (i >= kColumns) || 
                    ((i > j) && ((abs(RMatrixOP[i][j]) < kErrorThreshold) )) ||
                                ((i <= j));

          bool RIsNotFinite = (i < kColumns) && 
                                            !(std::isfinite(RMatrixOP[i][j]));

#else
          bool QREqA = (abs(AMatrix[matrix * kAMatrixSize + j * kRows + i].r()
                                                  - QRij.r()) < kErrorThreshold)
                    && (abs(AMatrix[matrix * kAMatrixSize + j * kRows + i].i() 
                                                - QRij.i()) < kErrorThreshold);

          bool QtQEqId = (
                    ((i == j) && (abs(QtQij.r() - 1) < QOrthoErrorThreshold)) ||
          (((i != j) || (j>=kRows)) && (abs(QtQij.r()) < QOrthoErrorThreshold))
                          ) && (abs(QtQij.i()) < QOrthoErrorThreshold);

          bool QQtEqId =  !squareMatrices || 
                          (
                            (
                    ((i == j) && (abs(QQtij.r() - 1) < QOrthoErrorThreshold)) || 
          (((i != j) || (j>=kRows)) && (abs(QQtij.r()) < QOrthoErrorThreshold))
                            )
                            && (abs(QQtij.i()) < QOrthoErrorThreshold)
                          );

          bool RIsUpperTriang = (i >= kColumns) || 
                                (
                                  (i > j) && 
                                  (
                                (abs(RMatrixOP[i][j].r()) < kErrorThreshold) && 
                                (abs(RMatrixOP[i][j].i()) < kErrorThreshold)
                                  )
                                )
                              || (i <= j);

          bool RIsNotFinite = (i < kColumns) && (!isFinite(RMatrixOP[i][j]));

#endif
          if (!QREqA ||
              !QtQEqId ||
              !QQtEqId ||
              !RIsUpperTriang||
              !isFinite(QRij)||
              !isFinite(QtQij)||
              !isFinite(QQtij)||
              RIsNotFinite
            ) {

            count++;

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
                        << " : " << QtQij << std::endl;  
              std::cout << "Q is not orthogonal at i=" << i << " j=" << j 
                        << " qtq=" << QtQij  
                        << " qqt=" << QQtij
                        << std::endl;             
              std::cout << "kQOrthoErrorThreshold=" << QOrthoErrorThreshold 
                        << std::endl;
              std::cout << "kErrorThreshold=" << kErrorThreshold 
                        << std::endl;           
            }
            if(!RIsUpperTriang) {
              std::cout  << "R is not upper triangular" << std::endl;             
            }
            if(!isFinite(QRij)) {
              std::cout << "QR[" << i << "][" << j << "] = " << QRij 
                        << " is not finite" << std::endl;
            }
            if(!isFinite(QtQij)) {
              std::cout << "QtQ[" << i << "][" << j << "] = " << QtQij 
                        << " is not finite" << std::endl;
            }
            if(RIsNotFinite) {
              std::cout << "R[" << i << "][" << j << "] = " << RMatrixOP[i][j] 
                        << " is not finite" << std::endl;
            }
            error = true;
          }
        }
      }

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
              << (((long long)matrices * (kAMatrixSize + kQRMatrixSize) *
              sizeof(float)) / pow(2, 30))
              << " GBs of memory was requested for " << matrices
              << " matrices, each of size " << kRows << " x "
              << kColumns << std::endl;
    std::terminate();
  }
}
