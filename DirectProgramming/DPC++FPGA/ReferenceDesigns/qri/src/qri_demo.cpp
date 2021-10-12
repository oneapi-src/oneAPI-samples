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
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl/INTEL/ac_types/ac_complex.hpp>
#include <chrono>
#include <list>

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
  returns a random floating-point value between min and max
*/
float randomValueInInterval(float min, float max){
  return min + static_cast <float>(rand()) 
                                      /(static_cast<float>(RAND_MAX)/(max-min));
}

/*
  QR decomposition of the input matrix
*/
template <int size, typename T>
void QRD(vector<T> &input, vector<T> &Q, vector<T> &R){

  // for (int i = -1; i < size; i++) {
  //   for (int j = i; j < size; j++) {

  //     T vector_t[size];
  //     T sori = s_or_i[j];

  //     for(int k=0; k<size; k++){
  //       vector_t[k] = input[j][k];
  //       if (j == i) {
  //         vector_ai[k] = vector_t[k];
  //       }
  //     }

  //     for(int k=0; k<size; k++){
  //       T sum_rhs = j == i ? {0} : vector_t[k];
  //       T mul_rhs = i < 0 ? {0} : sori + sum_rhs;
  //       vector_t[k] = vector_ai[k] * mul_rhs;
  //       if (i >= 0) {
  //         ap_matrix[j][k] = a_matrix[j][k] = vector_t[k];
  //       }
  //       if (j == i + 1) {
  //         vector_ti[k] = vector_t[k];
  //       }
  //     }

  //     T p_ij = {0};
  //     for(int k=0; k<size; k++){
  //       p_ij = p_ij + vector_t[k] * vector_ti[k];
  //     }

  //     if (j == i + 1) {
  //       p_ii_x = p_ij;
  //       i_r_ii_x = rsqrt(p_ij);
  //     }

  //     T s_ij = - p_ij / pip1

  //     if (j >= 0) {
  //       s_or_i[j] = j == i + 1 ? i_r_ii_x : s_ij; 
  //     }

  //     r_ii = j == i + 1 ? sycl::sqrt(p_ii_x) : i_r_ii_x * p_ij;

  //     if (j >= i + 1 && i + 1 < size) {
  //       R[i][j] = r_ii;
  //     }
  //   }
  // }
  int rows = size;
  int columns = size;
  for(int i=0; i<columns; i++){
    for(int j=0; j<columns; j++){
      R[i*columns + j] = 0;
    }
  }
  for(int i=0; i<rows; i++){
    for(int j=0; j<columns; j++){
      Q[i*columns + j] = 0;
    }
  }

  vector<T> S;
  S.resize(size*size);

  // <a_1, a_1>
  T a1a1 = 0;
  for(int k=0; k<rows; k++){
    T ak1 = input[k*columns + 0];
    a1a1 += ak1*ak1;
  }

  // P_{1,1} = <a_1, a_1>
  T pii = a1a1;

  // ir_{1,1} = 1/sqrt(p_{1,1})
  T irii = rsqrt(pii);

  // r_{1,1} = sqrt(p_{1,1})
  T rii = sqrt(pii);
  R[0*columns + 0] = rii;

  // for j=2:n do
  for(int j=1; j<rows; j++){
    // <a_1, a_j>
    T a1aj = 0;
    for(int k=0; k<rows; k++){
      // We should keep this matrix line a_1 localy as we reuse it many times
      T ak1 = input[(k*columns + 0)];
      T akj = input[(k*columns + j)];
      a1aj += ak1*akj;
    }  

    // p_{1,j} = <a_1, a_j>
    T p1j = a1aj;

    // s_{1,j} = p_{1,j} / p_{1,1}
    S[0*columns + j] = p1j / pii;

    // r_{1,j} = p_{1,j} * ir{1,1}
    R[0*columns + j] = p1j * irii;

  }

  // for i=1:n-1 do
  for(int i=0; i<rows-1; i++){
    // q_i = a_i * ir_{i,i}
    // out << "irii = " << irii << endl;
    for(int k=0; k<rows; k++){
      T aki = input[(k*columns + i)];
      // out << "A[" << i << "][" << k << "] = " << aki << endl;
      T akiirii = aki * irii;
      // out << "Q[" << i << "][" << k << "] = " << akiirii << endl;
      Q[(k*columns + i)] = akiirii;
    }  

    // for j=i+1:n do
    for(int j=i+1; j<rows; j++){
      // a_j = a_j - s_{i,j}*a_i
      for(int k=0; k<rows; k++){
        T sijaki = S[i*columns + j] * input[(k*columns + i)];
        T ajmsijaki = input[(k*columns + j)] - sijaki;
        input[(k*columns + j)] = ajmsijaki;
      }     

      // if j=i+1 then
      if(j == (i+1)){

        // <a_{i+1}, a_{i+1}>
        T ajaj = 0;
        for(int k=0; k<rows; k++){
          T akj = input[(k*columns + j)];
          ajaj += akj*akj;
        }  

        // p_{i+1, i+1} = <a_{i+1}, a_{i+1}>
        pii = ajaj;

        // ir_{j,j} = 1/sqrt(p_{j,j})
        irii = rsqrt(pii);

        // r_{i+1, i+1} = sqrt(p_{i+1, i+1})
        rii = sqrt(pii);
        R[j*columns + j] = rii;
      } 
      else{
        // We should keep aj locally because we reuse it every iteration of the loop
        // <a_{i+1}, aj>
        T aip1aj = 0;
        for(int k=0; k<rows; k++){
          T akip1 = input[(k*columns + (i+1))];
          T akj = input[(k*columns + j)];
          aip1aj += akip1*akj;
        }

        // P_{i+1, j} = <a_{i+1}, aj>
        T pip1j = aip1aj;

        // s_{i+1, j} = p_{i+1, j} / p_{i+1, i+1}
        S[(i+1)*columns + j] = pip1j / pii;

        // r_{i+1,j} = p{i+1, j} * ir_{i+1, i+1}
        R[(i+1)*columns + j] = pip1j * irii;
      } 
    }
  }

  // q_n = a_n * ir_{n,n}
  for(int k=0; k<rows; k++){
    T akn = input[k*columns + (rows-1)];
    Q[k*columns + (rows-1)] = akn * irii;
  }

}

/*
  Generate a random matrix with a given condition number
*/
template <int size, typename T>
void generateRandomMatrixWithGivenCondtitionNumber(float cn, vector<T> &output){

  constexpr float kRandomMin = -10;
  constexpr float kRandomMax = 10;

  // Start by generating two random matrices of size "size"
  vector<T> M1, M2;
  M1.resize(size*size);
  M2.resize(size*size);
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      float random1 = randomValueInInterval(kRandomMin, kRandomMax);
      M1[row*size + col] = random1;
      float random2 = randomValueInInterval(kRandomMin, kRandomMax);
      M2[row*size + col] = random2;
    }
  }

  // Get the QR decomposition of both matrices
  vector<T> Q1, R1;
  Q1.resize(size*size);
  R1.resize(size*size); 
  QRD<size>(M1, Q1, R1);

  vector<T> Q2, R2;
  Q2.resize(size*size);
  R2.resize(size*size); 
  QRD<size>(M2, Q2, R2);

  int j = size - 1;
  float l = pow(cn, 1.0/j);

  // Construct the singular value matrix l^0 = 1, then l^(k-1) = 1/cn
  vector<T> S;
  S.resize(size*size);
  
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      if(row == col){
        S[row*size + col] = pow(l, -row);
      }
      else{
        S[row*size + col] = 0.0;
      }
    }
  }

  // Compute the final matrix Q1*S*Q2
  // Start by computing Q1*S
  vector<T> Q1S;
  Q1S.resize(size*size);
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      T value = {0};
      for(int k=0; k<size; k++){
        value += Q1[row*size+k] * S[k*size+col];
      }
      Q1S[row*size+col] = value;
    }
  }
  // Then compute Q1S*Q2
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      T value = {0};
      for(int k=0; k<size; k++){
        value += Q1S[row*size+k] * Q2[k*size+col];
      }
      output[row*size+col] = value;
    }
  } 

  // Then compute Q1S*Q2
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      output[row*size+col] = output[row*size+col];
    }
  } 

}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr float kRandomMin = -2;
  constexpr float kRandomMax = 2;
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

    // For output-postprocessing
#if COMPLEX == 1
    ac_complex<float> inverse_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
#else
    float inverse_matrix_pp[ROWS_COMPONENT][COLS_COMPONENT];
#endif

    cout << "Generating " << matrices << " random matri"
         << ((matrices == 1) ? "x " : "ces ") << "\n";

    srand(kRandomSeed);

    // float maxConditionNumber = 0.0;
    int count = 0;
    for (size_t i = 0; i < matrices; i++) {
      // Generate a random matrix
      count++;
      // cout << "A MATRIX" << std::endl;
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          // Generate a random floating-point number between kRandomMin and
          // kRandomMax.
          float random_real = kRandomMin + static_cast <float>(rand())
                        /(static_cast<float>(RAND_MAX)/(kRandomMax-kRandomMin));

#if COMPLEX == 1
          float random_imag = kRandomMin + static_cast <float>(rand())
                        /(static_cast<float>(RAND_MAX)/(kRandomMax-kRandomMin));
          
          ac_complex<float> random_complex = {random_real, random_imag};

          A[i * kAMatrixSize + col * ROWS_COMPONENT + row] = random_complex;
#else
          A[i * kAMatrixSize + col * ROWS_COMPONENT + row] = random_real;
#endif
          // cout << A[i * kAMatrixSize + col * ROWS_COMPONENT + row] 
          //      << " ";
        }
        // cout << std::endl;
      }

      vector<float> test;
      test.resize(kAMatrixSize);

      generateRandomMatrixWithGivenCondtitionNumber<ROWS_COMPONENT>(3.2, test);

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          // A[i * kAMatrixSize + col * ROWS_COMPONENT + row] = random_real;
          A[i * kAMatrixSize + col * ROWS_COMPONENT + row] = test[row*COLS_COMPONENT + col];
        }
      }

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          std::cout << A[i * kAMatrixSize + row * COLS_COMPONENT + col] << " ";
        }
        std::cout << std::endl;
      }      

      // Check if the generated matrix is ill-conditioned for inversion
      // To do so, we compute de matrix inverse condition number:
      // norm_inf(A) * norm_inf(inv(A))
      // An approximation of inv(A) gives us enough information to 
      // compute the condition number

      // To compute the inverse of A, we use the Gaussian elimination 
      long double A_copy[COLS_COMPONENT][ROWS_COMPONENT];
      long double inverse[COLS_COMPONENT][ROWS_COMPONENT];

      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        for (size_t col = 0; col < COLS_COMPONENT; col++) {\
          if(row == col){
            // precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col] = 1.0;
            inverse[row][col] = 1.0;
          }
          else{
            // precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col] = 0.0;
            inverse[row][col] = 0.0;
          }
          A_copy[row][col] = test[row*COLS_COMPONENT + col];
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
        float pivot = A_copy[row][row];

        // If the pivot is zero, we need to swap the current row with 
        // another row that would give a non zero pivot.
        bool pivotIsZero = pivot == 0.0 || pivot == -0.0;
        if(pivotIsZero){ 
          for(int nextRow=row+1; nextRow<ROWS_COMPONENT; nextRow++){
            long double potentialPivotd = A_copy[nextRow][row];
            bool nextRowPivotIsZero = potentialPivotd == 0.0 || potentialPivotd == -0.0;
            // row can be used to swap
            if(!nextRowPivotIsZero){
              // We swap the two rows
              for(int j=0; j<COLS_COMPONENT; j++){
                long double tmp = A_copy[row][j];
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
        for(int rowToEliminate = ROWS_COMPONENT-1; rowToEliminate>=0; rowToEliminate--){
          if(rowToEliminate == row){
            continue;
          }

          long double factor = A_copy[rowToEliminate][row];
          for(int k=0; k<COLS_COMPONENT; k++){
            if(k == row){
              A_copy[rowToEliminate][k] = A_copy[rowToEliminate][k] - factor;
            }
            else{
              A_copy[rowToEliminate][k] = A_copy[rowToEliminate][k] - (A_copy[row][k] * factor);
            }
            inverse[rowToEliminate][k] = inverse[rowToEliminate][k] - (inverse[row][k] * factor);

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
      long double norm_inf_A = 0.0;
      long double norm_inf_inverse = 0.0;
      for (size_t row = 0; row < ROWS_COMPONENT; row++) {
        long double norm_i_A = 0.0;
        long double norm_i_inverse = 0.0;
        for (size_t col = 0; col < COLS_COMPONENT; col++) {
          // norm_i_A += abs(A[i * kAMatrixSize + col*COLS_COMPONENT + row]);
          norm_i_A += abs(test[row*COLS_COMPONENT + col]);
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
          precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col] = inverse[row][col];
        }
      }

      // cout << "A inverse matrix" << std::endl;
      // for (size_t row = 0; row < ROWS_COMPONENT; row++) {
      //   for (size_t col = 0; col < COLS_COMPONENT; col++) {
      //     cout << precomputed_inverse_matrix[i * kAMatrixSize + row*COLS_COMPONENT + col] << " ";
      //   }
      //   cout << std::endl;
      // }

      // Compute the confidition number
      float condition_number = norm_inf_A * norm_inf_inverse;
      cout << "norm_inf_A " << norm_inf_A << std::endl;
      cout << "norm_inf_inverse " << norm_inf_inverse << std::endl;
      cout << "condition_number " << condition_number << std::endl;

      // Regenerate this matrix if:
      // - the condition number is higher than the threshold
      // - we gave up on computing its inverse
      // if(std::log2(condition_number) > 10 || give_up){
      if(condition_number > 700 || give_up){
        i--;
      }
      else{
        // if(condition_number > maxConditionNumber){
        //   maxConditionNumber = condition_number;
        // }
        // cout << "Matrix " << counter << std::endl;
        // cout << "norm_inf_A " << norm_inf_A << std::endl;
        // cout << "norm_inf_inverse " << norm_inf_inverse << std::endl;
        // cout << "condition number: " << norm_inf_A * norm_inf_inverse << std::endl;
        cout << "matrix " << i << " found after " << count << " iterations " << std::endl;
        cout << "norm_inf_A " << norm_inf_A << std::endl;
        cout << "norm_inf_inverse " << norm_inf_inverse << std::endl;
        cout << "condition_number " << condition_number << std::endl;
        count =0;
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

    cout << "Verifying results on matrix";
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
            r_matrix_pp[i][j] = inverse_matrix[matrix * kInverseMatrixSize + idx];
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
          inverse_matrix_pp[i][j] = inverse_matrix[matrix * kInverseMatrixSize + idx];
          idx++;
        }
      }

      // for (size_t i = 0; i < ROWS_COMPONENT; i++) {
      //   for (size_t j = 0; j < COLS_COMPONENT; j++) {
      //     cout << inverse_matrix_pp[i][j] << " ";
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
            qr_ij[0] += inverse_matrix_pp[i][k].r() * r_matrix_pp[k][j].r() -
                        inverse_matrix_pp[i][k].i() * r_matrix_pp[k][j].i();
            qr_ij[1] += inverse_matrix_pp[i][k].r() * r_matrix_pp[k][j].i() +
                        inverse_matrix_pp[i][k].i() * r_matrix_pp[k][j].r();
          }

          qtq_ij[0] = 0;
          qtq_ij[1] = 0;
          if(i<COLS_COMPONENT){
            for (size_t k = 0; k < ROWS_COMPONENT; k++) {
              qtq_ij[0] += inverse_matrix_pp[k][i].r() * inverse_matrix_pp[k][j].r() +
                          inverse_matrix_pp[k][i].i() * inverse_matrix_pp[k][j].i();
              qtq_ij[1] += inverse_matrix_pp[k][i].r() * inverse_matrix_pp[k][j].i() -
                          inverse_matrix_pp[k][i].i() * inverse_matrix_pp[k][j].r();
            }
          }

          if(squareMatrices){
            qqt_ij[0] = 0;
            qqt_ij[1] = 0;
            if(i<COLS_COMPONENT){
              for (size_t k = 0; k < ROWS_COMPONENT; k++) {
                qqt_ij[0] += inverse_matrix_pp[i][k].r() * inverse_matrix_pp[j][k].r() +
                            inverse_matrix_pp[i][k].i() * inverse_matrix_pp[j][k].i();
                qqt_ij[1] += inverse_matrix_pp[i][k].r() * inverse_matrix_pp[j][k].i() -
                            inverse_matrix_pp[i][k].i() * inverse_matrix_pp[j][k].r();
              }
            }
          }

          bool qr_eq_a = (abs(A[matrix * kAMatrixSize +
                              j * ROWS_COMPONENT + i].r() - qr_ij[0]) 
                          < kErrorThreshold)
                      && (abs(A[matrix * kAMatrixSize +
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
                                  A[matrix * kAMatrixSize +
                                  j * ROWS_COMPONENT + i].r()
                                  << ", " <<
                                  A[matrix * kAMatrixSize +
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

      // idx = 0;
      for (size_t j = 0; j < COLS_COMPONENT; j++) {
        for (size_t i = 0; i < ROWS_COMPONENT; i++) {
          inverse_matrix_pp[j][i] = inverse_matrix[matrix * kInverseMatrixSize + idx];
          idx++;
        }
      }

      constexpr float kErrorThreshold = 1e-3;

      int kernelGreaterThanErrorThreshold = 0;
      double maxError = 0.0;

      for (size_t i = 0; i < ROWS_COMPONENT; i++) {
        for (size_t j = 0; j < COLS_COMPONENT; j++) {

          double kernelDiffDouble = abs(inverse_matrix_pp[i][j] - 
      precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j]);

          // unsigned int kernel_bits = *((unsigned int*)&(inverse_matrix_pp[i][j]));
          // unsigned int tb_bits = *((unsigned int*)&(precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j]));

          // unsigned int bits_diff; 
          // if(kernel_bits > tb_bits){
          //   bits_diff = kernel_bits - tb_bits;
          // }
          // else{
          //   bits_diff = tb_bits - kernel_bits;
          // }
          // if(bits_diff > max_bit_diff){
          //   cout << "new max kernel_bits diff "  << bits_diff << std::endl;
          //   cout << "kernel value "  << inverse_matrix_pp[i][j] << std::endl;
          //   cout << "tb value "  << precomputed_inverse_matrix[matrix * kAMatrixSize + i*COLS_COMPONENT + j] << std::endl;

          //   max_bit_diff = bits_diff;
          // }

          if(kernelDiffDouble > maxError){
            maxError = kernelDiffDouble;
          }

          if(kernelDiffDouble > kErrorThreshold){
            kernelGreaterThanErrorThreshold++;
          }
        }
      }
      
      std::cout << "Max error: " << maxError << std::endl; 

      if(maxError > maxErrorTotal){
        maxErrorTotal = maxError;
      }
      totalError += maxError;

      std::cout << "Kernel errors: " << kernelGreaterThanErrorThreshold << std::endl; 
      if(kernelGreaterThanErrorThreshold>0){
        error_count++;
        continue;
      }
#endif
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
