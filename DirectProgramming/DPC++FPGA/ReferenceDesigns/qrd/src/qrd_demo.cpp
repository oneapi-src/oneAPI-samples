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

void ComplexFloatQRDecomposition( vector<ac_complex<float>> &A_matrix, 
                      vector<ac_complex<float>> &QR_matrix,
                      queue &q, size_t matrices, size_t reps);

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

    vector<ac_complex<float>> a_matrix;
    vector<ac_complex<float>> qr_matrix;

    a_matrix.resize(matrices * kAMatrixSizeFactor);
    qr_matrix.resize(matrices * kQRMatrixSize);

    // For output-postprocessing
    float q_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT][2];
    float r_matrix_pp[COLS_COMPONENT][COLS_COMPONENT][2];

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(kRandomSeed);

    for (size_t i = 0; i < matrices; i++) {
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          int val = rand();
          float random_real = val % (kRandomMax - kRandomMin) + kRandomMin;
          val = rand();
          float random_imag = val % (kRandomMax - kRandomMin) + kRandomMin;
          
          ac_complex<float> random_complex = {random_real, random_imag};

          a_matrix[i * kAMatrixSizeFactor + col * ROWS_COMPONENT + row] = 
                                                                random_complex;
        }
      }
    }

    // Accelerator warmup
    ComplexFloatQRDecomposition(a_matrix, qr_matrix, q, 1, 1); 

#if defined(FPGA_EMULATOR)
    size_t reps = 2;
#else
    size_t reps = 32;
#endif
    cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    ComplexFloatQRDecomposition(a_matrix, qr_matrix, q, matrices, reps);
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

    for (size_t matrix : to_check) {
      cout << " " << matrix << std::endl;
      size_t idx = 0;
      for (size_t i = 0; i < COLS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          if (j < i)
            r_matrix_pp[i][j][0] = r_matrix_pp[i][j][1] = 0;
          else {
            r_matrix_pp[i][j][0] = qr_matrix[matrix * kQRMatrixSize + idx].r();
            r_matrix_pp[i][j][1] = qr_matrix[matrix * kQRMatrixSize + idx].i();
            idx++;
          }
        }
      }

      // idx = 0;
      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          q_matrix_pp[i][j][0] = qr_matrix[matrix * kQRMatrixSize + idx].r();
          q_matrix_pp[i][j][1] = qr_matrix[matrix * kQRMatrixSize + idx].i();
          idx++;

          // cout << q_matrix[matrix * kQMatrixSize + idx] << " ";
        }
        // cout << std::endl;
      }

      constexpr float kErrorThreshold = 1e-4;
      size_t count = 0;
      bool error = false;
      float qr_ij[2] = {0};
      float qtq_ij[2] = {0};
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          qr_ij[0] = 0;
          qr_ij[1] = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            qr_ij[0] += q_matrix_pp[i][k][0] * r_matrix_pp[k][j][0] -
                        q_matrix_pp[i][k][1] * r_matrix_pp[k][j][1];
            qr_ij[1] += q_matrix_pp[i][k][0] * r_matrix_pp[k][j][1] +
                        q_matrix_pp[i][k][1] * r_matrix_pp[k][j][0];
          }

          qtq_ij[0] = 0;
          qtq_ij[1] = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            qtq_ij[0] += q_matrix_pp[i][k][0] * q_matrix_pp[j][k][0] +
                        q_matrix_pp[i][k][1] * q_matrix_pp[j][k][1];
            qtq_ij[1] += q_matrix_pp[i][k][0] * q_matrix_pp[j][k][1] -
                        q_matrix_pp[i][k][1] * q_matrix_pp[j][k][0];
          }


          bool qr_eq_a = (abs(a_matrix[matrix * kAMatrixSizeFactor +
                              j * ROWS_COMPONENT + i].r() - qr_ij[0]) 
                          < kErrorThreshold)
                      && (abs(a_matrix[matrix * kAMatrixSizeFactor +
                              j * ROWS_COMPONENT + i].i() - qr_ij[1]) 
                          < kErrorThreshold);


          bool qtq_ortho = (((i == j) && (abs(qtq_ij[0] - 1) < kErrorThreshold))
                        || ((i != j) && (abs(qtq_ij[0]) < kErrorThreshold)))
                        && (abs(qtq_ij[1]) < kErrorThreshold);

          bool r_upper_triang = ((i > j) && 
                              ((abs(r_matrix_pp[i][j][0]) < kErrorThreshold) &&
                                (abs(r_matrix_pp[i][j][1]) < kErrorThreshold)))
                                || ((i <= j));

          if (!qr_eq_a || !qtq_ortho || !r_upper_triang
              || !std::isfinite(qr_ij[0])
              || !std::isfinite(qr_ij[1]) 
              || !std::isfinite(qtq_ij[0])
              || !std::isfinite(qtq_ij[1])
              || !std::isfinite(r_matrix_pp[i][j][0])
              || !std::isfinite(r_matrix_pp[i][j][1])
            ) {

            count++;

            if(error){
              continue;
            }

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
            if(!qtq_ortho) {
              cout  << "Q is not orthogonal" << std::endl;             
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
            if(!std::isfinite(r_matrix_pp[i][j][0]) || 
               !std::isfinite(r_matrix_pp[i][j][1])) {
              cout  << "R[" << i << "][" << j << "] = (" << r_matrix_pp[i][j][0] 
                    << ", " << r_matrix_pp[i][j][1] << ") is not finite" \
                    << std::endl;
            }
            error = true;
          }
        }
      }

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
