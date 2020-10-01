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
#include <chrono>
#include <list>

#include "qrd.hpp"

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

// Header locations and some DPC++ extensions changed between beta09 and beta10
// Temporarily modify the code sample to accept either version
#define BETA09 20200827
#if __SYCL_COMPILER_VERSION <= BETA09
  #include <CL/sycl/intel/fpga_extensions.hpp>
  namespace INTEL = sycl::intel;  // Namespace alias for backward compatibility
#else
  #include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace std;
using namespace std::chrono;
using namespace sycl;

// Run the modified Gram-Schmidt QR Decomposition algorithm on the given
// matrices. The function will do the following:
//   1. Transfer the input matrices to the FPGA.
//   2. Run the algorithm.
//   3. Copy the output data back to host device.
// The above process is carried out 'reps' number of times.
void QRDecomposition(vector<float> &in_matrix, vector<float> &out_matrix, queue &q,
                size_t matrices, size_t reps);

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRandomMin = 1;
  constexpr size_t kRandomMax = 10;

  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    cout << "Must run at least 1 matrix\n";
    return 1;
  }

  try {
#if defined(FPGA_EMULATOR)
    INTEL::fpga_emulator_selector device_selector;
#else
    INTEL::fpga_selector device_selector;
#endif

    queue q = queue(device_selector, dpc_common::exception_handler);
    device device = q.get_device();
    cout << "Device name: " << device.get_info<info::device::name>().c_str()
         << "\n";

    vector<float> a_matrix;
    vector<float> qr_matrix;

    constexpr size_t kAMatrixSizeFactor = ROWS_COMPONENT * COLS_COMPONENT * 2;
    constexpr size_t kQRMatrixSizeFactor =
        (ROWS_COMPONENT + 1) * COLS_COMPONENT * 3;
    constexpr size_t kIndexAccessFactor = 2;

    a_matrix.resize(matrices * kAMatrixSizeFactor);
    qr_matrix.resize(matrices * kQRMatrixSizeFactor);

    // For output-postprocessing
    float q_matrix[ROWS_COMPONENT][COLS_COMPONENT][2];
    float r_matrix[COLS_COMPONENT][COLS_COMPONENT][2];

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(kRandomSeed);

    for (size_t i = 0; i < matrices; i++) {
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          int random_val = rand();
          float random_double =
              random_val % (kRandomMax - kRandomMin) + kRandomMin;
          a_matrix[i * kAMatrixSizeFactor +
                   col * ROWS_COMPONENT * kIndexAccessFactor +
                   row * kIndexAccessFactor] = random_double;
          int random_val_imag = rand();
          random_double =
              random_val_imag % (kRandomMax - kRandomMin) + kRandomMin;
          a_matrix[i * kAMatrixSizeFactor +
                   col * ROWS_COMPONENT * kIndexAccessFactor +
                   row * kIndexAccessFactor + 1] = random_double;
        }
      }
    }

    QRDecomposition(a_matrix, qr_matrix, q, 1, 1);  // Accelerator warmup

#if defined(FPGA_EMULATOR)
    size_t reps = 2;
#else
    size_t reps = 32;
#endif
    cout << "Running QR decomposition of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    high_resolution_clock::time_point start_time = high_resolution_clock::now();
    QRDecomposition(a_matrix, qr_matrix, q, matrices, reps);
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
      cout << " " << matrix;
      size_t idx = 0;
      for (size_t i = 0; i < COLS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          if (j < i)
            r_matrix[i][j][0] = r_matrix[i][j][1] = 0;
          else {
            r_matrix[i][j][0] = qr_matrix[matrix * kQRMatrixSizeFactor + idx++];
            r_matrix[i][j][1] = qr_matrix[matrix * kQRMatrixSizeFactor + idx++];
          }
        }
      }

      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          q_matrix[i][j][0] = qr_matrix[matrix * kQRMatrixSizeFactor + idx++];
          q_matrix[i][j][1] = qr_matrix[matrix * kQRMatrixSizeFactor + idx++];
        }
      }

      float acc_real = 0;
      float acc_imag = 0;
      float v_matrix[ROWS_COMPONENT][COLS_COMPONENT][2] = {{{0}}};
      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {
          acc_real = 0;
          acc_imag = 0;
          for (size_t k = 0; k < COLS_COMPONENT; k++) {
            acc_real += q_matrix[i][k][0] * r_matrix[k][j][0] -
                        q_matrix[i][k][1] * r_matrix[k][j][1];
            acc_imag += q_matrix[i][k][0] * r_matrix[k][j][1] +
                        q_matrix[i][k][1] * r_matrix[k][j][0];
          }
          v_matrix[i][j][0] = acc_real;
          v_matrix[i][j][1] = acc_imag;
        }
      }

      float error = 0;
      size_t count = 0;
      constexpr float kErrorThreshold = 1e-4;
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          if (std::isnan(v_matrix[row][col][0]) ||
              std::isnan(v_matrix[row][col][1])) {
            count++;
          }
          float real = v_matrix[row][col][0] -
                       a_matrix[matrix * kAMatrixSizeFactor +
                                col * ROWS_COMPONENT * kIndexAccessFactor +
                                row * kIndexAccessFactor];
          float imag = v_matrix[row][col][1] -
                       a_matrix[matrix * kAMatrixSizeFactor +
                                col * ROWS_COMPONENT * kIndexAccessFactor +
                                row * kIndexAccessFactor + 1];
          if (sqrt(real * real + imag * imag) >= kErrorThreshold) {
            error += sqrt(real * real + imag * imag);
            count++;
          }
        }
      }

      if (count > 0) {
        cout << "\nFAILED\n";
        cout << "\n"
             << "!!!!!!!!!!!!!! Error = " << error << " in " << count << " / "
             << ROWS_COMPONENT * COLS_COMPONENT << "\n";
        return 1;
      }
    }

    cout << "\nPASSED\n";
    return 0;

  } catch (sycl::exception const &e) {
    cout << "Caught a synchronous SYCL exception: " << e.what() << "\n";
    cout << "   If you are targeting an FPGA hardware, "
            "ensure that your system is plugged to an FPGA board that is "
            "set up correctly"
         << "\n";
    cout << "   If you are targeting the FPGA emulator, compile with "
            "-DFPGA_EMULATOR"
         << "\n";

    terminate();
  }
}
