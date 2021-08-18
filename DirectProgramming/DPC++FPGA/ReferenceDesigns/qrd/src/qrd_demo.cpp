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
void ComplexFloatQRDecomposition( vector<ac_complex<float>> &A_matrix, 
                                  vector<ac_complex<float>> &QR_matrix,
                                  queue &q, size_t matrices, size_t reps);
#else
void FloatQRDecomposition(  vector<float> &A_matrix, 
                            vector<float> &QR_matrix,
                            queue &q, size_t matrices, size_t reps);
#endif

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;
  constexpr size_t kAMatrixSizeFactor = ROWS_COMPONENT * COLS_COMPONENT;
  
  constexpr size_t kQMatrixSize = ROWS_COMPONENT * COLS_COMPONENT;
  constexpr size_t kRMatrixSize = COLS_COMPONENT * (COLS_COMPONENT + 1) / 2;
  constexpr size_t kQRMatrixSize = kQMatrixSize + kRMatrixSize;

  constexpr size_t kIndexAccessFactor = 2;

  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    cout << "Must run at least 1 matrix\n";
    return 1;
  }

  try {
#if defined(FPGA_EMULATOR)
    ext::intel::fpga_emulator_selector device_selector;
#else
    ext::intel::fpga_selector device_selector;
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
    vector<ac_complex<float>> a_matrix;
    vector<ac_complex<float>> qr_matrix;
#else
    vector<float> a_matrix;
    vector<float> qr_matrix; 
#endif

    a_matrix.resize(matrices * kAMatrixSizeFactor);
    qr_matrix.resize(matrices * kQRMatrixSize);

    // For output-postprocessing
#if COMPLEX == 1
    ac_complex<float> q_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
    ac_complex<float> r_matrix_pp[COLS_COMPONENT][COLS_COMPONENT];
#else
    float q_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
    float r_matrix_pp[COLS_COMPONENT][COLS_COMPONENT];
#endif

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

            a_matrix[i * kAMatrixSizeFactor + col * ROWS_COMPONENT + row] = 
                                                                  random_complex;
  #else
            // if(test == 60){
            a_matrix[i * kAMatrixSizeFactor + col * ROWS_COMPONENT + row] = 
                                                                  random_real;
  #endif
            // cout << a_matrix[i * kAMatrixSizeFactor + col * ROWS_COMPONENT + row] 
            //      << " ";
            // }
          }
          // cout << std::endl;
        }
      }

    // }

    // Accelerator warmup
// #if COMPLEX == 1
//     ComplexFloatQRDecomposition(a_matrix, qr_matrix, q, 1, 1); 
// #else
//     FloatQRDecomposition(a_matrix, qr_matrix, q, 1, 1); 
// #endif

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
    ComplexFloatQRDecomposition(a_matrix, qr_matrix, q, matrices, reps);
#else
    FloatQRDecomposition(a_matrix, qr_matrix, q, matrices, reps);
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

    // for (int i=0; i<matrices; i++){
    //   to_check.push_back(i);
    // }

    cout << "Verifying results on matrix";

    bool squareMatrices = ROWS_COMPONENT == COLS_COMPONENT;

    for (size_t matrix : to_check) {
      cout << " " << matrix << std::endl;
      size_t idx = 0;

#if COMPLEX == 1
      // cout << "R MATRIX" << std::endl;
      for (size_t i = 0; i < COLS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          if (j < i)
            r_matrix_pp[i][j] = {0};
          else {
            r_matrix_pp[i][j] = qr_matrix[matrix * kQRMatrixSize + idx];
            idx++;
          }
          // cout << r_matrix_pp[i][j] << " ";
        }
        // cout << std::endl;
      }

      // idx = 0;
      // cout << "Q MATRIX" << std::endl;
      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          q_matrix_pp[i][j] = qr_matrix[matrix * kQRMatrixSize + idx];
          idx++;
        }
      }

      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     cout << q_matrix_pp[i][j] << " ";
      //   }
      //   cout << std::endl;
      // }

      constexpr float kErrorThreshold = 1e-4;

      float QOrthoErrorThreshold = pow(2.0, -9);
      size_t count = 0;
      bool error = false;
      float qr_ij[2] = {0};
      float qtq_ij[2] = {0};
      float qqt_ij[2] = {0};
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          qr_ij[0] = 0;
          qr_ij[1] = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            qr_ij[0] += q_matrix_pp[i][k].r() * r_matrix_pp[k][j].r() -
                        q_matrix_pp[i][k].i() * r_matrix_pp[k][j].i();
            qr_ij[1] += q_matrix_pp[i][k].r() * r_matrix_pp[k][j].i() +
                        q_matrix_pp[i][k].i() * r_matrix_pp[k][j].r();
          }

          qtq_ij[0] = 0;
          qtq_ij[1] = 0;
          if(i<COLS_COMPONENT){
            for (size_t k = 0; k < ROWS_COMPONENT; k++) {
              qtq_ij[0] += q_matrix_pp[k][i].r() * q_matrix_pp[k][j].r() +
                          q_matrix_pp[k][i].i() * q_matrix_pp[k][j].i();
              qtq_ij[1] += q_matrix_pp[k][i].r() * q_matrix_pp[k][j].i() -
                          q_matrix_pp[k][i].i() * q_matrix_pp[k][j].r();
            }
          }

          if(squareMatrices){
            qqt_ij[0] = 0;
            qqt_ij[1] = 0;
            if(i<COLS_COMPONENT){
              for (size_t k = 0; k < ROWS_COMPONENT; k++) {
                qqt_ij[0] += q_matrix_pp[i][k].r() * q_matrix_pp[j][k].r() +
                            q_matrix_pp[i][k].i() * q_matrix_pp[j][k].i();
                qqt_ij[1] += q_matrix_pp[i][k].r() * q_matrix_pp[j][k].i() -
                            q_matrix_pp[i][k].i() * q_matrix_pp[j][k].r();
              }
            }
          }

          bool qr_eq_a = (abs(a_matrix[matrix * kAMatrixSizeFactor +
                              j * ROWS_COMPONENT + i].r() - qr_ij[0]) 
                          < kErrorThreshold)
                      && (abs(a_matrix[matrix * kAMatrixSizeFactor +
                              j * ROWS_COMPONENT + i].i() - qr_ij[1]) 
                          < kErrorThreshold);


          bool qtq_is_id = 
                      (((i == j) && (abs(qtq_ij[0] - 1) < QOrthoErrorThreshold))
            || (((i != j) || (j>=ROWS_COMPONENT)) 
                                    && (abs(qtq_ij[0]) < QOrthoErrorThreshold)))
                                    && (abs(qtq_ij[1]) < QOrthoErrorThreshold);

          bool qqt_is_id = !squareMatrices ||
                    ((((i == j) && (abs(qqt_ij[0] - 1) < QOrthoErrorThreshold))
            || (((i != j) || (j>=ROWS_COMPONENT)) 
                                    && (abs(qqt_ij[0]) < QOrthoErrorThreshold)))
                                    && (abs(qqt_ij[1]) < QOrthoErrorThreshold));

          bool r_upper_triang = (i >= COLS_COMPONENT) || 
                              ((i > j) && 
                              ((abs(r_matrix_pp[i][j].r()) < kErrorThreshold) &&
                                (abs(r_matrix_pp[i][j].i()) < kErrorThreshold)))
                              || ((i <= j));

          bool r_is_not_finite = (i < COLS_COMPONENT) && 
                                  (!(std::isfinite(r_matrix_pp[i][j].r())) ||
                                    !(std::isfinite(r_matrix_pp[i][j].i())));

          if (!qr_eq_a 
              || !qtq_is_id 
              || !qqt_is_id 
              || !r_upper_triang
              || !std::isfinite(qr_ij[0])
              || !std::isfinite(qr_ij[1]) 
              || !std::isfinite(qtq_ij[0])
              || !std::isfinite(qtq_ij[1])
              || !std::isfinite(qqt_ij[1])
              || !std::isfinite(qqt_ij[1])
              || r_is_not_finite
            ) {

            count++;

            if(error){
              continue;
            }
            cout << "Error at i= " << i << " j= " << j << std::endl;

            if(!qr_eq_a){
              cout  << "Error: A[" << i << "][" << j << "] = (" << 
                                  a_matrix[matrix * kAMatrixSizeFactor +
                                  j * ROWS_COMPONENT + i].r()
                                  << ", " <<
                                  a_matrix[matrix * kAMatrixSizeFactor +
                                  j * ROWS_COMPONENT + i].i()
                    << ") but QR[" << i << "][" << j << "] = (" << qr_ij[0] 
                    << ", " << qr_ij[1] << ")" << std::endl;
            }
            if(!qr_eq_a) {
              cout  << "The difference is greater than tolerated (" 
                    << kErrorThreshold << ")" << std::endl;
            }
            if(!qtq_is_id || !qqt_is_id) {
              cout  << "Q is not orthogonal at i=" << i << " j=" << j 
              << " qtq=(" << qtq_ij[0] << ", " << qtq_ij[1] << ")"  
              << " qqt=(" << qqt_ij[0] << ", " << qqt_ij[1] << ")"
              << std::endl;             
              cout << "kQOrthoErrorThreshold=" << QOrthoErrorThreshold 
                    << std::endl;
              cout << "kErrorThreshold=" << kErrorThreshold 
                    << std::endl;
            }
            if(!r_upper_triang) {
              cout  << "R is not upper triangular" << std::endl;             
            }
            if(!std::isfinite(qr_ij[0]) || !std::isfinite(qr_ij[1])) {
              cout  << "QR[" << i << "][" << j << "] = (" << qr_ij[0] << ", " 
                    << qr_ij[1] << ") is not finite" << std::endl;
            }
            if(!std::isfinite(qtq_ij[0]) || !std::isfinite(qtq_ij[1])) {
              cout  << "QtQ[" << i << "][" << j << "] = (" << qtq_ij[0] << ", " 
                    << qtq_ij[1] << ") is not finite" << std::endl;
            }
            if(r_is_not_finite) {
              cout  << "R[" << i << "][" << j << "] = (" 
                    << r_matrix_pp[i][j].r() 
                    << ", " << r_matrix_pp[i][j].i() << ") is not finite"
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
            r_matrix_pp[i][j] = 0;
          else {
            r_matrix_pp[i][j] = qr_matrix[matrix * kQRMatrixSize + idx];
            idx++;
          }
          // cout << r_matrix_pp[i][j] << " ";
        }
        // cout << std::endl;
      }
      
      // cout << std::endl;

      // idx = 0;
      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          q_matrix_pp[i][j] = qr_matrix[matrix * kQRMatrixSize + idx];
          idx++;
        }
      }

      // cout << "Q MATRIX" << std::endl;
      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     cout << q_matrix_pp[i][j] << " ";
      //   }
      //   cout << std::endl;
      // }

      constexpr float kErrorThreshold = 1e-4;
      float QOrthoErrorThreshold = pow(2.0, -9);
      size_t count = 0;
      bool error = false;
      float qr_ij = 0;
      float qtq_ij = 0;
      float qqt_ij = 0;
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          qr_ij = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            qr_ij += q_matrix_pp[i][k] * r_matrix_pp[k][j];
          }

          qtq_ij = 0;
          if(i<COLS_COMPONENT){
            for (size_t k = 0; k < ROWS_COMPONENT; k++) {
              qtq_ij += q_matrix_pp[k][i] * q_matrix_pp[k][j];
            }
          }

          if(squareMatrices){
            qqt_ij = 0;
            if(i<COLS_COMPONENT){
              for (size_t k = 0; k < ROWS_COMPONENT; k++) {
                qqt_ij += q_matrix_pp[i][k] * q_matrix_pp[j][k];
              }
            }
          }

          bool qr_eq_a = (abs(a_matrix[matrix * kAMatrixSizeFactor +
                              j * ROWS_COMPONENT + i] - qr_ij) 
                          < kErrorThreshold);

          bool qtq_is_id = 
                      ((i == j) && (abs(qtq_ij - 1) < QOrthoErrorThreshold)) ||
                           ((i != j) && (abs(qtq_ij) < QOrthoErrorThreshold));

          bool qqt_is_id = !squareMatrices || 
                      (((i == j) && (abs(qqt_ij - 1) < QOrthoErrorThreshold)) ||
                           ((i != j) && (abs(qqt_ij) < QOrthoErrorThreshold)));

          bool r_upper_triang = (i >= COLS_COMPONENT) || 
                      ((i > j) && ((abs(r_matrix_pp[i][j]) < kErrorThreshold) ))
                                || ((i <= j));

          bool r_is_not_finite = (i < COLS_COMPONENT) && 
                                            !(std::isfinite(r_matrix_pp[i][j]));
          if (!qr_eq_a 
              || !qtq_is_id
              || !qqt_is_id
              || !r_upper_triang
              || !std::isfinite(qr_ij)
              || !std::isfinite(qtq_ij)
              || !std::isfinite(qqt_ij)
              || r_is_not_finite
            ) {

            count++;

            if(error){
              continue;
            }

            if(!qr_eq_a){
              cout  << "Error: A[" << i << "][" << j << "] = " << 
                                  a_matrix[matrix * kAMatrixSizeFactor +
                                  j * ROWS_COMPONENT + i]
                    << " but QR[" << i << "][" << j << "] = " << qr_ij 
                    << std::endl;
            }
            if(!qr_eq_a) {
              cout  << "The difference is greater than tolerated (" 
                    << kErrorThreshold << ")" << std::endl;
            }
            if(!qtq_is_id || !qqt_is_id) {
              cout  << "Q is not orthogonal at i " << i << " j " << j << " : " 
                    << qtq_ij << std::endl;  

              cout  << "Q is not orthogonal at i=" << i << " j=" << j 
              << " qtq=" << qtq_ij  
              << " qqt=(" << qqt_ij
              << std::endl;             
              cout << "kQOrthoErrorThreshold=" << QOrthoErrorThreshold 
                    << std::endl;
              cout << "kErrorThreshold=" << kErrorThreshold 
                    << std::endl;           
            }
            if(!r_upper_triang) {
              cout  << "R is not upper triangular" << std::endl;             
            }
            if(!std::isfinite(qr_ij)) {
              cout  << "QR[" << i << "][" << j << "] = " << qr_ij 
                    << " is not finite" << std::endl;
            }
            if(!std::isfinite(qtq_ij)) {
              cout  << "QtQ[" << i << "][" << j << "] = " << qtq_ij 
                    << " is not finite" << std::endl;
            }
            if(r_is_not_finite) {
              cout  << "R[" << i << "][" << j << "] = " << r_matrix_pp[i][j] 
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
         << (((long long)matrices * (kAMatrixSizeFactor + kQRMatrixSize) *
              sizeof(float)) /
             pow(2, 30))
         << " GB of memory was requested for " << matrices
         << " matrices, each of size " << ROWS_COMPONENT << " x "
         << COLS_COMPONENT << "\n";

    terminate();
  }
}
