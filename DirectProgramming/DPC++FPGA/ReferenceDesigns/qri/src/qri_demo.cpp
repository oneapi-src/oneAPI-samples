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
#include <iomanip> 

// dpc_common.hpp can be found in the dev-utilities include folder.
// e.g., $ONEAPI_ROOT/dev-utilities//include/dpc_common.hpp
#include "dpc_common.hpp"

using namespace std;
using namespace std::chrono;
using namespace sycl;

#if COMPLEX == 1
void ComplexFloatQRI( vector<ac_complex<float>> &A, 
                                  vector<ac_complex<float>> &inverse_matrix,
                                  queue &q, size_t matrices, size_t reps);
#else
void FloatQRI(  vector<float> &A, 
                            vector<float> &inverse_matrix,
                            queue &q, size_t matrices, size_t reps);
#endif

/*
  Returns a random floating-point value between min and max
*/
float randomValueInInterval(float min, float max){
  return min + static_cast <float>(rand()) 
                                      /(static_cast<float>(RAND_MAX)/(max-min));
}

/*
  Generate a random matrix M with a given epsilon such that
  cond(M, inf) <= (1+epsilon)/(1-epsilon)

  Algorithm courtesy of Carl Christian Kjelgaard Mikkelsen 

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
  % The matrix A is not necessarily well-conditioned, but the matrix B
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
void generateMatrixWithCondititionNumber(float epsilon, vector<T> &output){

  constexpr float kRandomMin = 0;
  constexpr float kRandomMax = 1;

  // Start by generating a random matrices R with diagonal elements set to 0
  // and measuring the weights of the off diagonal entries
  vector<T> R, weights;
  R.resize(size*size);
  weights.resize(size);
  for(int row=0; row<size; row++){
    weights[row] = {0};
    for(int col=0; col<size; col++){
      if(col != row){
        float random1 = randomValueInInterval(kRandomMin, kRandomMax);
        T elem;
  #if COMPLEX == 1
        float random1I = randomValueInInterval(kRandomMin, kRandomMax);
        elem = {random1, random1I};
        R[row*size + col] = elem;
  #else
        elem = random1;
        R[row*size + col] = elem;
  #endif
        weights[row] += elem;
      }
    }
    // construct the new diagonal element
    weights[row] /= epsilon; 
    R[row*size + row] = weights[row];
  }

  // Now we need to do the diagonal scaling by solving:
  // diag(diag(A))*output = A
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      output[row*size + col] = R[row*size + col] / R[row*size + row];
    }
  }

}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kAMatrixSize = ROWS_COMPONENT * COLS_COMPONENT;
  constexpr size_t kInverseMatrixSize = ROWS_COMPONENT * COLS_COMPONENT;

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
    vector<ac_complex<float>> A;
    vector<ac_complex<float>> inverse_matrix;
    vector<ac_complex<float>> precomputed_inverse_matrix; 
#else
    vector<float> A;
    vector<float> inverse_matrix; 
    vector<float> precomputed_inverse_matrix; 
#endif

    A.resize(matrices * kAMatrixSize);
    inverse_matrix.resize(matrices * kInverseMatrixSize);
    precomputed_inverse_matrix.resize(matrices * kInverseMatrixSize);

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(kRandomSeed);

    for (size_t i = 0; i < matrices; i++) {
      // Generate a random matrix

#if COMPLEX == 1
      vector<ac_complex<float>> randomRealMatrix;
#else
      vector<float> randomRealMatrix;
#endif
      randomRealMatrix.resize(kAMatrixSize);

      generateMatrixWithCondititionNumber<ROWS_COMPONENT>(0.5, 
                                                              randomRealMatrix);

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          A[i * kAMatrixSize + col * ROWS_COMPONENT + row] = 
                                   {randomRealMatrix[row*COLS_COMPONENT + col]};
        }
      }

      // Check if the generated matrix is ill-conditioned for inversion
      // To do so, we compute de matrix inverse condition number:
      // norm_inf(A) * norm_inf(inv(A))
      // An approximation of inv(A) gives us enough information to 
      // compute the condition number

      // To compute the inverse of A, we use the Gaussian elimination 
#if COMPLEX == 1
      ac_complex<double> A_copy[COLS_COMPONENT][ROWS_COMPONENT];
      ac_complex<double> inverse[COLS_COMPONENT][ROWS_COMPONENT];
#else
      double A_copy[COLS_COMPONENT][ROWS_COMPONENT];
      double inverse[COLS_COMPONENT][ROWS_COMPONENT];
#endif

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {\
          if(row == col){
            inverse[row][col] = {1.0};
          }
          else{
            inverse[row][col] = {0.0};
          }
          A_copy[row][col] = randomRealMatrix[row*COLS_COMPONENT + col];
          // A_copy[row][col] = A[i * kAMatrixSize + col*COLS_COMPONENT + row];
        }
      }

      // cout << "A_copy matrix" << std::endl;
      // for (size_t row = 0; row < ROWS_COMPONENT; row++) {
      //   for (size_t col = 0; col < COLS_COMPONENT; col++) {
      //     cout << A_copy[row][col] << " ";
      //   }
      //   cout << std::endl;
      // }

      // If we can't find a solution using the gaussian elimination, 
      // we may give up on this matrix and generate another one
      bool give_up = false;
     
      for (int row = 0; row < ROWS_COMPONENT; row++){
        // Find the next pivot
        auto pivot = A_copy[row][row];

        // If the pivot is zero, we need to swap the current row with 
        // another row that would give a non zero pivot.
        bool pivotIsZero = pivot == 0.0 || pivot == -0.0;
        if(pivotIsZero){ 
          for(int nextRow=row+1; nextRow<ROWS_COMPONENT; nextRow++){
            auto potentialPivotd = A_copy[nextRow][row];
            bool nextRowPivotIsZero = potentialPivotd == 0.0 
                                   || potentialPivotd == -0.0;
            // row can be used to swap
            if(!nextRowPivotIsZero){
              // We swap the two rows
              for(int j=0; j<COLS_COMPONENT; j++){
                auto tmp = A_copy[row][j];
                A_copy[row][j] = A_copy[nextRow][j];
                A_copy[nextRow][j] = tmp;

                tmp = inverse[row][j];
                inverse[row][j] = inverse[nextRow][j];
                inverse[nextRow][j] = tmp;
                // tmp = precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + j];
                // precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + j] = precomputed_inverse_matrix[i * kAMatrixSize + nextRow*COLS_COMPONENT + j];
                // precomputed_inverse_matrix[i * kAMatrixSize + nextRow*COLS_COMPONENT + j] = tmp;
              }
              break;
            }
          }
          // Get the new pivot
          pivot = A_copy[row][row];

          // If we were not able to find a pivot, give up on this matrix
          give_up = pivot == 0.0 || pivot == -0.0;
          if(give_up){
            break;
          }
        }

        // Divide the current row by the pivot value
        for(int k = 0; k < COLS_COMPONENT; k++) {
          A_copy[row][k] = A_copy[row][k]/pivot;
          // precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + k] = precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + k]/pivot;
          inverse[row][k] = inverse[row][k]/pivot;
        }

        // Eliminate the current row in all other rows
        for(int rowToEliminate = ROWS_COMPONENT-1; rowToEliminate>=0; 
                                                              rowToEliminate--){
          if(rowToEliminate == row){
            continue;
          }

          auto factor = A_copy[rowToEliminate][row];
          for(int k=0; k<COLS_COMPONENT; k++){
            if(k == row){
              A_copy[rowToEliminate][k] = A_copy[rowToEliminate][k] - factor;
            }
            else{
              A_copy[rowToEliminate][k] = A_copy[rowToEliminate][k] 
                                          - (A_copy[row][k] * factor);
            }
            inverse[rowToEliminate][k] = inverse[rowToEliminate][k] 
                                        - (inverse[row][k] * factor);

            // precomputed_inverse_matrix[i * kAMatrixSize + rowToEliminate*COLS_COMPONENT + k] = 
            //       precomputed_inverse_matrix[i * kAMatrixSize + rowToEliminate*COLS_COMPONENT + k]
            //       - (precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + k] * factor);
          }
        }

        // if(row == 1 || row == 2){
          // cout << "A inverse matrix at row " << row << std::endl;
          // for (size_t row = 0; row < ROWS_COMPONENT; row++) {
          //   // for (size_t col = 0; col < COLS_COMPONENT; col++) {
          //   //   cout << A_copy[row][col] << " ";
          //   // }
          //   //   cout << "   ";
          //   for (size_t col = 0; col < COLS_COMPONENT; col++) {
          //     // cout << precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col] << " ";
          //     cout << inverse[row][col] << " ";
          //   }

          //   cout << std::endl;
          // } 
        // }

      }

      // Compute the norm inf of both the input and the inverse matrices
      // to compute the condition number
      double norm_inf_A = 0.0;
      double norm_inf_inverse = 0.0;
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        double norm_i_A = 0.0;
        double norm_i_inverse = 0.0;
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          // norm_i_A += abs(A[i * kAMatrixSize + col*COLS_COMPONENT + row]);
          norm_i_A += abs(randomRealMatrix[row*COLS_COMPONENT + col]);
          // norm_i_inverse += abs(precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col]);
          norm_i_inverse += abs(inverse[row][col]);
        }
        if(norm_i_A>norm_inf_A){
          norm_inf_A = norm_i_A;
        }
        if(norm_i_inverse>norm_inf_inverse){
          norm_inf_inverse = norm_i_inverse;
        }
      }

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
#if COMPLEX == 1
          if(!std::isfinite(inverse[row][col].r()) || 
            !std::isfinite(inverse[row][col].i()))
#else
          if(!std::isfinite(inverse[row][col]))
#endif       
          {
            give_up = true;
          }
          else{
            precomputed_inverse_matrix[i * kAMatrixSize + 
                                  row*COLS_COMPONENT + col] = inverse[row][col];
          }
        }
      }

      // Compute the confidition number
      float condition_number = norm_inf_A * norm_inf_inverse;

      // Regenerate this matrix if:
      // - the condition number is higher than the threshold
      // - we gave up on computing its inverse
      if(condition_number > 8 || give_up){
        i--;
      }
      else{
        // std::cout << "A matrix" << std::endl;
        // for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        //   for (size_t col = 0; col < COLS_COMPONENT; col++) {
        //     std::cout << std::setprecision(3) << A[i * kAMatrixSize + col * COLS_COMPONENT + row] << " ";
        //   }
        //   std::cout << std::endl;
        // }      

        cout << "norm_inf_A " << norm_inf_A << std::endl;
        cout << "norm_inf_inverse " << norm_inf_inverse << std::endl;
        cout << "condition_number " << condition_number << std::endl;
        // if(i ==36)
        //     exit(0);
      }

    }

    // cout << "Max condition number: " << maxConditionNumber << std::endl;


#if defined(FPGA_EMULATOR)
    size_t reps = 1;
#else
    size_t reps = 32;
    // Accelerator warmup
#if COMPLEX == 1
    ComplexFloatQRI(A, inverse_matrix, q, 1, 1); 
#else
    FloatQRI(A, inverse_matrix, q, 1, 1);
#endif
#endif
    cout << "Running QR inversion of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    high_resolution_clock::time_point start_time = high_resolution_clock::now();
#if COMPLEX == 1
    ComplexFloatQRI(A, inverse_matrix, q, matrices, reps);
#else
    FloatQRI(A, inverse_matrix, q, matrices, reps);
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
    // to_check.push_back(0);
    // Spot check the last and the middle one
    // if (matrices > 2) to_check.push_back(matrices / 2);
    // if (matrices > 1) to_check.push_back(matrices - 1);

    for (int i=0; i<matrices; i++){
      to_check.push_back(i);
    }

    int error_count = 0;
    double maxErrorTotal = 0.0;
    double totalError = 0.0;

    // For output-postprocessing
#if COMPLEX == 1
    ac_complex<float> inverse_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
#else
    float inverse_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
#endif

    constexpr float kErrorThreshold = 1e-3;
    
    cout << "Verifying results on matrix";
    for (size_t matrix : to_check) {
      cout << " " << matrix << std::endl;
      size_t idx = 0;

      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          inverse_matrix_pp[j][i] = 
                              inverse_matrix[matrix * kInverseMatrixSize + idx];
          idx++;
        }
      }

      // std::cout << "Kernel inverse" << std::endl;
      // for(int row=0; row<ROWS_COMPONENT; row++){
      //   for(int col=0; col<COLS_COMPONENT; col++){
      //     std::cout << inverse_matrix_pp[row][col] << " ";
      //   }
      //   std::cout << std::endl;
      // }

      // std::cout << "Precomputed inverse" << std::endl;
      // for(int row=0; row<ROWS_COMPONENT; row++){
      //   for(int col=0; col<COLS_COMPONENT; col++){
      //     std::cout << precomputed_inverse_matrix[matrix * kAMatrixSize + row*COLS_COMPONENT + col] << " ";
      //   }
      //   std::cout << std::endl;
      // }

      int kernelGreaterThanErrorThreshold = 0;
      double maxError = 0.0;

#if COMPLEX == 1

      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {

          double diffR = abs(inverse_matrix_pp[i][j].r() - 
  precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j].r());

          double diffI = abs(inverse_matrix_pp[i][j].i() - 
  precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j].i());

          if(!std::isfinite(diffR) || !std::isfinite(diffR)){
            kernelGreaterThanErrorThreshold++;
          }

          if(diffR > maxError){
            maxError = diffR;
          }
          if(diffI > maxError){
            maxError = diffI;
          }

          if(diffR > kErrorThreshold){
            kernelGreaterThanErrorThreshold++;
          }
          if(diffI > kErrorThreshold){
            kernelGreaterThanErrorThreshold++;
          }
        }
      }
      
#else

      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {

          double diff = abs(inverse_matrix_pp[i][j] - 
      precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j]);

          if(!std::isfinite(diff)){
            kernelGreaterThanErrorThreshold++;
          }

          if(diff > maxError){
            maxError = diff;
          }

          if(diff > kErrorThreshold){
            kernelGreaterThanErrorThreshold++;
          }
        }
      }
      
#endif

      std::cout << "Max error: " << maxError << std::endl; 

      if(maxError > maxErrorTotal){
        maxErrorTotal = maxError;
      }
      totalError += maxError;

      std::cout << "Kernel errors: " << kernelGreaterThanErrorThreshold 
                << std::endl; 
      if(kernelGreaterThanErrorThreshold>0){
        error_count++;
        break;
      }
    }

    cout << "maxErrorTotal " << maxErrorTotal << std::endl;
    cout << "average max error " << totalError/to_check.size() << std::endl;
    if (error_count > 0) {
      cout << "\nFAILED\n";
      cout << "\n"
           << "!!!!!!!!!!!!!! " << error_count << " errors" 
           << std::endl;
      return 1;
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
         << (((long long)matrices * (kAMatrixSize + kInverseMatrixSize) *
              sizeof(float)) /
             pow(2, 30))
         << " GBs of memory was requested for " << matrices
         << " matrices, each of size " << ROWS_COMPONENT << " x "
         << COLS_COMPONENT << "\n";

    terminate();
  }
}
