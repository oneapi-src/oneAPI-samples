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

using namespace std;
using namespace std::chrono;
using namespace sycl;


#if COMPLEX == 1
void ComplexFloatQRDecomposition( vector<ac_complex<float>> &AMatrix, 
                                  vector<ac_complex<float>> &QMatrix,
                                  vector<ac_complex<float>> &RMatrix,
                                  queue &q, size_t matrices, size_t reps);
#else
void FloatQRDecomposition(  vector<float> &AMatrix, 
                            vector<float> &QMatrix,
                            vector<float> &RMatrix,
                            queue &q, size_t matrices, size_t reps);
#endif

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kAMatrixSize = ROWS_COMPONENT * COLS_COMPONENT;
  
  constexpr size_t kQMatrixSize = ROWS_COMPONENT * COLS_COMPONENT;
  constexpr size_t kRMatrixSize = COLS_COMPONENT * (COLS_COMPONENT + 1) / 2;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;

  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    cout << "Must run at least 1 matrix\n";
    return 1;
  }

  try {
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    queue q = queue(device_selector, dpc_common::exception_handler);
    device device = q.get_device();
    cout << "Device name: " << device.get_info<info::device::name>().c_str()
         << "\n";

#if COMPLEX == 1
  cout << "Type is complex" << std::endl;
#else
  cout << "Type is not complex" << std::endl;
#endif

#if COMPLEX == 1
    vector<ac_complex<float>> AMatrix;
    vector<ac_complex<float>> QMatrix;
    vector<ac_complex<float>> RMatrix;
#else
    vector<float> AMatrix;
    vector<float> QMatrix; 
    vector<float> RMatrix; 
#endif

    AMatrix.resize(matrices * kAMatrixSize);
    QMatrix.resize(matrices * kQMatrixSize);
    RMatrix.resize(matrices * kRMatrixSize);

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(kRandomSeed);
    // for (int test=0; test<61; test++){

    for (size_t i = 0; i < matrices; i++) {
      // cout << "A MATRIX" << std::endl;
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          int val = rand();
          float random_real = val % (kRandomMax - kRandomMin) + kRandomMin;

#if COMPLEX == 1
          val = rand();
          float random_imag = val % (kRandomMax - kRandomMin) + kRandomMin;
          
          ac_complex<float> random_complex = {random_real, random_imag};

          AMatrix[i * kAMatrixSize + col * ROWS_COMPONENT + row] = 
                                                                random_complex;
#else
          // if(test == 60){
          AMatrix[i * kAMatrixSize + col * ROWS_COMPONENT + row] = 
                                                                random_real;
#endif
          // cout << AMatrix[i * kAMatrixSize + col * ROWS_COMPONENT + row] 
          //      << " ";
          // }
        }
        // cout << std::endl;
      }
    }

    // }


#if defined(FPGA_EMULATOR)
#else
    // Accelerator warmup
#if COMPLEX == 1
    ComplexFloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, 1, 1); 
#else
    FloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, 1, 1); 
#endif
#endif

#if defined(FPGA_EMULATOR)
    size_t reps = 1;
#else
    size_t reps = 32;
#endif
    cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    high_resolution_clock::time_point start_time = high_resolution_clock::now();
#if COMPLEX == 1
    ComplexFloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, matrices, reps);
#else
    FloatQRDecomposition(AMatrix, QMatrix, RMatrix, q, matrices, reps);
#endif
    high_resolution_clock::time_point end_time = high_resolution_clock::now();
    duration<double> diff = end_time - start_time;
    q.throw_asynchronous();

    cout << "   Total duration:   " << diff.count() << " s"
         << "\n";
    cout << "Throughput: " << reps * matrices / diff.count() / 1000
         << "k matrices/s"
         << "\n";

    list<size_t> to_check;
    // We will check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (matrices > 2) to_check.push_back(matrices / 2);
    if (matrices > 1) to_check.push_back(matrices - 1);

    cout << "Verifying results on matrix";


    // For output-postprocessing (OP)
#if COMPLEX == 1
    ac_complex<float> QMatrixOP[ROWS_COMPONENT][COLS_COMPONENT];
    ac_complex<float> RMatrixOP[COLS_COMPONENT][COLS_COMPONENT];
#else
    float QMatrixOP[ROWS_COMPONENT][COLS_COMPONENT];
    float RMatrixOP[COLS_COMPONENT][COLS_COMPONENT];
#endif

    bool squareMatrices = ROWS_COMPONENT == COLS_COMPONENT;

    for (size_t matrix : to_check) {
      cout << " " << matrix << std::endl;
      size_t RIdx = 0;
      size_t QIdx = 0;

#if COMPLEX == 1
      // cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < COLS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          if (j < i)
            RMatrixOP[i][j] = {0};
          else {
            RMatrixOP[i][j] = RMatrix[matrix * kRMatrixSize + RIdx];
            RIdx++;
          }
          // cout << RMatrixOP[i][j] << " ";
        }
        // cout << std::endl;
      }

      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          QMatrixOP[i][j] = QMatrix[matrix * kQMatrixSize + QIdx];
          QIdx++;
        }
      }

      // cout << "Q MATRIX" << std::endl;
      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     cout << QMatrixOP[i][j] << " ";
      //   }
      //   cout << std::endl;
      // }

      constexpr float kErrorThreshold = 1e-4;

      float QOrthoErrorThreshold = pow(2.0, -9);
      size_t count = 0;
      bool error = false;
      // Q * R
      float QRij[2] = {0};
      // transpose(Q) * Q
      float QtQij[2] = {0};
      // Q * transpose(Q)
      float QQtij[2] = {0};
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          QRij[0] = 0;
          QRij[1] = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            QRij[0] += QMatrixOP[i][k].r() * RMatrixOP[k][j].r() -
                        QMatrixOP[i][k].i() * RMatrixOP[k][j].i();
            QRij[1] += QMatrixOP[i][k].r() * RMatrixOP[k][j].i() +
                        QMatrixOP[i][k].i() * RMatrixOP[k][j].r();
          }

          QtQij[0] = 0;
          QtQij[1] = 0;
          if(i<COLS_COMPONENT){
            for (size_t k = 0; k < ROWS_COMPONENT; k++) {
              QtQij[0] += QMatrixOP[k][i].r() * QMatrixOP[k][j].r() +
                          QMatrixOP[k][i].i() * QMatrixOP[k][j].i();
              QtQij[1] += QMatrixOP[k][i].r() * QMatrixOP[k][j].i() -
                          QMatrixOP[k][i].i() * QMatrixOP[k][j].r();
            }
          }

          if(squareMatrices){
            QQtij[0] = 0;
            QQtij[1] = 0;
            if(i<COLS_COMPONENT){
              for (size_t k = 0; k < ROWS_COMPONENT; k++) {
                QQtij[0] += QMatrixOP[i][k].r() * QMatrixOP[j][k].r() +
                            QMatrixOP[i][k].i() * QMatrixOP[j][k].i();
                QQtij[1] += QMatrixOP[i][k].r() * QMatrixOP[j][k].i() -
                            QMatrixOP[i][k].i() * QMatrixOP[j][k].r();
              }
            }
          }

          bool QREqA = (abs(AMatrix[matrix * kAMatrixSize +
                              j * ROWS_COMPONENT + i].r() - QRij[0]) 
                          < kErrorThreshold)
                      && (abs(AMatrix[matrix * kAMatrixSize +
                              j * ROWS_COMPONENT + i].i() - QRij[1]) 
                          < kErrorThreshold);


          bool QtQEqId = 
                      (((i == j) && (abs(QtQij[0] - 1) < QOrthoErrorThreshold))
            || (((i != j) || (j>=ROWS_COMPONENT)) 
                                    && (abs(QtQij[0]) < QOrthoErrorThreshold)))
                                    && (abs(QtQij[1]) < QOrthoErrorThreshold);

          bool QQtEqId = !squareMatrices ||
                    ((((i == j) && (abs(QQtij[0] - 1) < QOrthoErrorThreshold))
            || (((i != j) || (j>=ROWS_COMPONENT)) 
                                    && (abs(QQtij[0]) < QOrthoErrorThreshold)))
                                    && (abs(QQtij[1]) < QOrthoErrorThreshold));

          bool RIsUpperTriang = (i >= COLS_COMPONENT) || 
                              ((i > j) && 
                              ((abs(RMatrixOP[i][j].r()) < kErrorThreshold) &&
                                (abs(RMatrixOP[i][j].i()) < kErrorThreshold)))
                              || ((i <= j));

          bool RIsNotFinite = (i < COLS_COMPONENT) && 
                                  (!(std::isfinite(RMatrixOP[i][j].r())) ||
                                    !(std::isfinite(RMatrixOP[i][j].i())));

          if (!QREqA 
              || !QtQEqId 
              || !QQtEqId 
              || !RIsUpperTriang
              || !std::isfinite(QRij[0])
              || !std::isfinite(QRij[1]) 
              || !std::isfinite(QtQij[0])
              || !std::isfinite(QtQij[1])
              || !std::isfinite(QQtij[1])
              || !std::isfinite(QQtij[1])
              || RIsNotFinite
            ) {

            count++;

            if(error){
              continue;
            }
            cout << "Error at i= " << i << " j= " << j << std::endl;

            if(!QREqA){
              cout  << "Error: A[" << i << "][" << j << "] = (" << 
                                  AMatrix[matrix * kAMatrixSize +
                                  j * ROWS_COMPONENT + i].r()
                                  << ", " <<
                                  AMatrix[matrix * kAMatrixSize +
                                  j * ROWS_COMPONENT + i].i()
                    << ") but QR[" << i << "][" << j << "] = (" << QRij[0] 
                    << ", " << QRij[1] << ")" << std::endl;
            }
            if(!QREqA) {
              cout  << "The difference is greater than tolerated (" 
                    << kErrorThreshold << ")" << std::endl;
            }
            if(!QtQEqId || !QQtEqId) {
              cout  << "Q is not orthogonal at i=" << i << " j=" << j 
              << " qtq=(" << QtQij[0] << ", " << QtQij[1] << ")"  
              << " qqt=(" << QQtij[0] << ", " << QQtij[1] << ")"
              << std::endl;             
              cout << "kQOrthoErrorThreshold=" << QOrthoErrorThreshold 
                    << std::endl;
              cout << "kErrorThreshold=" << kErrorThreshold 
                    << std::endl;
            }
            if(!RIsUpperTriang) {
              cout  << "R is not upper triangular" << std::endl;             
            }
            if(!std::isfinite(QRij[0]) || !std::isfinite(QRij[1])) {
              cout  << "QR[" << i << "][" << j << "] = (" << QRij[0] << ", " 
                    << QRij[1] << ") is not finite" << std::endl;
            }
            if(!std::isfinite(QtQij[0]) || !std::isfinite(QtQij[1])) {
              cout  << "QtQ[" << i << "][" << j << "] = (" << QtQij[0] << ", " 
                    << QtQij[1] << ") is not finite" << std::endl;
            }
            if(RIsNotFinite) {
              cout  << "R[" << i << "][" << j << "] = (" 
                    << RMatrixOP[i][j].r() 
                    << ", " << RMatrixOP[i][j].i() << ") is not finite"
                    << std::endl;
            }
            error = true;
          }
        }
      }
#else

      // cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < COLS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          if (j < i)
            RMatrixOP[i][j] = 0;
          else {
            RMatrixOP[i][j] = RMatrix[matrix * kRMatrixSize + RIdx];
            RIdx++;
          }
          // cout << RMatrixOP[i][j] << " ";
        }
        // cout << std::endl;
      }
      // cout << std::endl;

      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          QMatrixOP[i][j] = QMatrix[matrix * kQMatrixSize + QIdx];
          QIdx++;
        }
      }

      // cout << "Q MATRIX" << std::endl;
      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     cout << QMatrixOP[i][j] << " ";
      //   }
      //   cout << std::endl;
      // }

      constexpr float kErrorThreshold = 1e-4;
      float QOrthoErrorThreshold = pow(2.0, -9);
      size_t count = 0;
      bool error = false;
      float QRij = 0;
      float QtQij = 0;
      float QQtij = 0;
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          QRij = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            QRij += QMatrixOP[i][k] * RMatrixOP[k][j];
          }

          QtQij = 0;
          if(i<COLS_COMPONENT){
            for (size_t k = 0; k < ROWS_COMPONENT; k++) {
              QtQij += QMatrixOP[k][i] * QMatrixOP[k][j];
            }
          }

          if(squareMatrices){
            QQtij = 0;
            if(i<COLS_COMPONENT){
              for (size_t k = 0; k < ROWS_COMPONENT; k++) {
                QQtij += QMatrixOP[i][k] * QMatrixOP[j][k];
              }
            }
          }

          bool QREqA = (abs(AMatrix[matrix * kAMatrixSize +
                              j * ROWS_COMPONENT + i] - QRij) 
                          < kErrorThreshold);

          bool QtQEqId = 
                      ((i == j) && (abs(QtQij - 1) < QOrthoErrorThreshold)) ||
                           ((i != j) && (abs(QtQij) < QOrthoErrorThreshold));

          bool QQtEqId = !squareMatrices || 
                      (((i == j) && (abs(QQtij - 1) < QOrthoErrorThreshold)) ||
                           ((i != j) && (abs(QQtij) < QOrthoErrorThreshold)));

          bool RIsUpperTriang = (i >= COLS_COMPONENT) || 
                      ((i > j) && ((abs(RMatrixOP[i][j]) < kErrorThreshold) ))
                                || ((i <= j));

          bool RIsNotFinite = (i < COLS_COMPONENT) && 
                                            !(std::isfinite(RMatrixOP[i][j]));
          if (!QREqA 
              || !QtQEqId
              || !QQtEqId
              || !RIsUpperTriang
              || !std::isfinite(QRij)
              || !std::isfinite(QtQij)
              || !std::isfinite(QQtij)
              || RIsNotFinite
            ) {

            count++;

            if(error){
              continue;
            }

            if(!QREqA){
              cout  << "Error: A[" << i << "][" << j << "] = " << 
                                  AMatrix[matrix * kAMatrixSize +
                                  j * ROWS_COMPONENT + i]
                    << " but QR[" << i << "][" << j << "] = " << QRij 
                    << std::endl;
            }
            if(!QREqA) {
              cout  << "The difference is greater than tolerated (" 
                    << kErrorThreshold << ")" << std::endl;
            }
            if(!QtQEqId || !QQtEqId) {
              cout  << "Q is not orthogonal at i " << i << " j " << j << " : " 
                    << QtQij << std::endl;  

              cout  << "Q is not orthogonal at i=" << i << " j=" << j 
              << " qtq=" << QtQij  
              << " qqt=(" << QQtij
              << std::endl;             
              cout << "kQOrthoErrorThreshold=" << QOrthoErrorThreshold 
                    << std::endl;
              cout << "kErrorThreshold=" << kErrorThreshold 
                    << std::endl;           
            }
            if(!RIsUpperTriang) {
              cout  << "R is not upper triangular" << std::endl;             
            }
            if(!std::isfinite(QRij)) {
              cout  << "QR[" << i << "][" << j << "] = " << QRij 
                    << " is not finite" << std::endl;
            }
            if(!std::isfinite(QtQij)) {
              cout  << "QtQ[" << i << "][" << j << "] = " << QtQij 
                    << " is not finite" << std::endl;
            }
            if(RIsNotFinite) {
              cout  << "R[" << i << "][" << j << "] = " << RMatrixOP[i][j] 
                    << " is not finite" << std::endl;
            }
            error = true;
          }
        }
      }
#endif

      if (count > 0) {
        cout << "\nFAILED\n";
        cout << "\n"
             << "!!!!!!!!!!!!!! " << count << " errors" 
             << std::endl;
        return 1;
      }
    }

    cout << "\nPASSED\n";
    return 0;

  } catch (sycl::exception const &e) {
    cerr << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    cerr << "   If you are targeting an FPGA hardware, "
            "ensure that your system is plugged to an FPGA board that is "
            "set up correctly"
         << "\n";
    cerr << "   If you are targeting the FPGA emulator, compile with "
            "-DFPGA_EMULATOR"
         << "\n";

    terminate();
  } catch (std::bad_alloc const &e) {
    cerr << "Caught a memory allocation exception on the host: " << e.what()
         << "\n";
    cerr << "   You can reduce the memory requirement by reducing the number "
            "of matrices generated. Specify a smaller number when running the "
            "executable."
         << "\n";
    cerr << "   In this run, more than "
         << (((long long)matrices * (kAMatrixSize + kQRMatrixSize) *
              sizeof(float)) /
             pow(2, 30))
         << " GB of memory was requested for " << matrices
         << " matrices, each of size " << ROWS_COMPONENT << " x "
         << COLS_COMPONENT << "\n";

    terminate();
  }
}
