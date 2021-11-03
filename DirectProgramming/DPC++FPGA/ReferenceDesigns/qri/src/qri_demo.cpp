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
#include "qri.hpp"

/*
  COMPLEX, COLS_COMPONENT, ROWS_COMPONENT, FIXED_ITERATIONS_QRD and 
  FIXED_ITERATIONS_QRI are defined by the build system.
  Depending on the value of COMPLEX, the real or complex QR based matrix 
  inversion function (QRI) is defined.

  Each matrix (input and output) are represented using vectors and are 
  interpreted in a column fashion (transposed).

  Function arguments:
  - AMatrix:    The input matrix to be inverted. 
                Interpreted as a transposed matrix.
  - invMatrix:  The output matrix. The function will overwrite this matrix.
                Will contain the inverse of AMatrix.
  - q:          The device queue.
  - matrices:   The number of matrices to be processed.
                The input matrices are read sequentially from the AMatrix 
                vector.
  - reps:       The number of repetitions of the computation to execute.
                (for performance evaluation)
*/
#if COMPLEX == 0
// Real single precision floating-point QR based inversion
void QRI( std::vector<float> &AMatrix, 
          std::vector<float> &invMatrix,
          sycl::queue &q, 
          size_t matrices, 
          size_t reps) {

  constexpr bool isComplex = false;
  QRI_impl< COLS_COMPONENT, 
            ROWS_COMPONENT, 
            FIXED_ITERATIONS_QRD, 
            FIXED_ITERATIONS_QRI, 
            isComplex, 
            float>(AMatrix, invMatrix, q, matrices, reps); 
}
#else
// Complex single precision floating-point QR based inversion
void QRI( std::vector<ac_complex<float>> &AMatrix, 
          std::vector<ac_complex<float>> &invMatrix,
          sycl::queue &q, 
          size_t matrices, 
          size_t reps) {

  constexpr bool isComplex = true;
  QRI_impl< COLS_COMPONENT, 
            ROWS_COMPONENT, 
            FIXED_ITERATIONS_QRD, 
            FIXED_ITERATIONS_QRI, 
            isComplex, 
            float>(AMatrix, invMatrix, q, matrices, reps); 
}
#endif


/*
  Returns a random floating-point value between min and max
*/
float randomValueInInterval(float min, float max){
  return min + static_cast <float>(rand()) 
                                      /(static_cast<float>(RAND_MAX)/(max-min));
}

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
void generateMatrixWithCondititionNumber(float epsilon, std::vector<T> &output){

  // Random min and max values for the random floating-point value generation  
  constexpr float kRandomMin = 0;
  constexpr float kRandomMax = 1;

  // Generate a random matrix R with diagonal elements set to 0
  // and measure the weights of the off diagonal entries
  std::vector<T> R, weights;
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

    // Construct the new diagonal element
    weights[row] /= epsilon; 
    R[row*size + row] = weights[row];
  }

  // Perform the diagonal scaling by solving:
  // diag(diag(A))*output = A
  for(int row=0; row<size; row++){
    for(int col=0; col<size; col++){
      output[row*size + col] = R[row*size + col] / R[row*size + row];
    }
  }
}

int main(int argc, char *argv[]) {
  constexpr size_t kRandomSeed = 1138;
  constexpr size_t kRows = ROWS_COMPONENT;
  constexpr size_t kColumns = COLS_COMPONENT;
  constexpr size_t kAMatrixSize = kRows * kColumns;
  constexpr size_t kInverseMatrixSize = kRows * kColumns;
  constexpr bool kComplex = COMPLEX != 0;

  // Get the number of random matrices to decompose from the command line
  // If no value is given, will only decompose 1 random matrix
  size_t matrices = argc > 1 ? atoi(argv[1]) : 1;
  if (matrices < 1) {
    std::cout << "Must run at least 1 matrix\n";
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
                                                                float>::type TF;
    // Select a type for computing the inverse in the testbench using a more 
    // precise format than the kernel
    typedef typename std::conditional<kComplex, ac_complex<double>, 
                                                              double>::type TD;

    // Create vectors to hold all the input and output matrices
    std::vector<TF> A;
    std::vector<TF> invMatrix;
    std::vector<TF> precomputedInvMatrix; 

    A.resize(matrices * kAMatrixSize);
    invMatrix.resize(matrices * kInverseMatrixSize);
    precomputedInvMatrix.resize(matrices * kInverseMatrixSize);

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

    // Generate the random input matrices and precompute their inverse
    srand(kRandomSeed);
    for (size_t i = 0; i < matrices; i++) {

      std::vector<TF> randomMatrix;
      randomMatrix.resize(kAMatrixSize);
      // Setting an epsilon of 0.5 ensures that the inverse matrix will have
      // a condition number using the infinite norm lower than 1.5/0.5 = 3
      float epsilon = 0.5;
      generateMatrixWithCondititionNumber<kRows>(epsilon, randomMatrix);

      // Copy the generated matrix in the A vector
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          A[i * kAMatrixSize + col * kRows + row] = 
                                              randomMatrix[row*kColumns + col];
        }
      }

      // Precompute the inverse of A using the Gaussian elimination 
      // A copy of A that will be modified
      TD ACopy[kColumns][kRows];
      // The inverse matrix that will be iteratively constructed starting from
      // the identity matrix 
      TD inverse[kColumns][kRows];

      // Copy A in ACopy and set "inverse" to the identity matrix
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {\
          if(row == col){
            inverse[row][col] = {1.0};
          }
          else{
            inverse[row][col] = {0.0};
          }
          ACopy[row][col] = randomMatrix[row*kColumns + col];
        }
      }

      // If we can't find a solution using the Gaussian elimination, 
      // we may give up on this matrix and generate another one
      bool give_up = false;
     
      // Perform the Gaussian elimination
      for (int row = 0; row < kRows; row++){
        // Find the next pivot
        auto pivot = ACopy[row][row];

        // If the pivot is zero, we need to swap the current row with 
        // another row that would give a non-zero pivot.
        bool pivotIsZero = pivot == 0.0 || pivot == -0.0;
        if(pivotIsZero){ 
          // Find an alternate row to use for pivoting
          for(int nextRow=row+1; nextRow<kRows; nextRow++){
            TD potentialPivot = ACopy[nextRow][row];
            bool potentialPivotIsZero = potentialPivot == 0.0 || 
                                        potentialPivot == -0.0;
            // row can be used to swap
            if(!potentialPivotIsZero){
              // Swap the two rows
              for(int j=0; j<kColumns; j++){
                auto tmp = ACopy[row][j];
                ACopy[row][j] = ACopy[nextRow][j];
                ACopy[nextRow][j] = tmp;

                tmp = inverse[row][j];
                inverse[row][j] = inverse[nextRow][j];
                inverse[nextRow][j] = tmp;
              }

              // The swap was successful, stop searching for a row to swap with
              break;
            }
          }

          // Get the new pivot
          pivot = ACopy[row][row];

          // If the swapping was unsuccessful are the new pivot is 0, 
          // give up on this matrix generate another one
          give_up = pivot == 0.0 || pivot == -0.0;
          if(give_up){
            break;
          }
        }

        // Divide the current row by the pivot value
        for(int k = 0; k < kColumns; k++) {
          ACopy[row][k] = ACopy[row][k]/pivot;
          inverse[row][k] = inverse[row][k]/pivot;
        }

        // Eliminate the current row in all other rows
        for(int rowToEliminate = kRows-1; rowToEliminate>=0; rowToEliminate--){
          if(rowToEliminate == row){
            continue;
          }

          auto factor = ACopy[rowToEliminate][row];
          for(int k=0; k<kColumns; k++){
            if(k == row){
              ACopy[rowToEliminate][k] = ACopy[rowToEliminate][k] - factor;
            }
            else{
              ACopy[rowToEliminate][k] = ACopy[rowToEliminate][k] 
                                          - (ACopy[row][k] * factor);
            }
            inverse[rowToEliminate][k] = inverse[rowToEliminate][k] 
                                        - (inverse[row][k] * factor);
          }
        }
      }

      // Compute the norm inf of both the input and the inverse matrices
      // to compute the condition number and verify that it is lower than the
      // expected threshold
      double normInfA = 0.0;
      double normInfInverse = 0.0;
      for (size_t row = 0; row < kRows; row++) {
        // Compute the norm inf of the current row on both matrices
        double normCurrentRowOfA = 0.0;
        double normCurrentRowOfInverse = 0.0;
        for (size_t col = 0; col < kColumns; col++) {
          normCurrentRowOfA += abs(randomMatrix[row*kColumns + col]);
          normCurrentRowOfInverse += abs(inverse[row][col]);
        }

        // Update the norm inf of both matrices if the norm inf of the current
        // row is the new max 
        if(normCurrentRowOfA > normInfA){
          normInfA = normCurrentRowOfA;
        }
        if(normCurrentRowOfInverse > normInfInverse){
          normInfInverse = normCurrentRowOfInverse;
        }
      }

      // Copy the current inverse matrix in the precomputedInvMatrix vector
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {
          // If any of the element in not finite, give up on this matrix
          if(!isFinite(inverse[row][col])){
            give_up = true;
          }
          else{
            precomputedInvMatrix[i * kAMatrixSize + row*kColumns + col] =
                                                             inverse[row][col];
          }
        }
      }

      // Compute the condition number
      double condition_number = normInfA * normInfInverse;
      double expectedConditionNumber = (1 + epsilon)/(1 - epsilon);

      // Regenerate this matrix if:
      // - the condition number is higher than the expected one
      // - we gave up earlier
      if(condition_number > expectedConditionNumber || give_up){
        i--;
      }
#ifdef DEBUG
      else{
        std::cout << "A matrix" << std::endl;
        for (size_t row = 0; row < kRows; row++) {
          for (size_t col = 0; col < kColumns; col++) {
            std::cout << A[i * kAMatrixSize + col * kColumns + row] << " ";
          }
          std::cout << std::endl;
        }      
        std::cout << "normInfA " << normInfA << std::endl;
        std::cout << "normInfInverse " << normInfInverse << std::endl;
        std::cout << "condition_number " << condition_number << std::endl;
      }
#endif
    }

#if defined(FPGA_EMULATOR)
    size_t reps = 1;
#else
    size_t reps = 32;
    // Accelerator warmup
    QRI(A, invMatrix, q, 2048, 1); 
#endif

    std::cout << "Running QR inversion of " << matrices << " matri"
         << ((matrices == 1) ? "x " : "ces ")
         << ((reps > 1) ? "repeatedly" : "") << "\n";

    // Launch the compute kernel and time the execution
    std::chrono::high_resolution_clock::time_point start_time = 
                                      std::chrono::high_resolution_clock::now();
    QRI(A, invMatrix, q, matrices, reps);
    std::chrono::high_resolution_clock::time_point end_time = 
                                      std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    q.throw_asynchronous();

    std::cout << "   Total duration:   " << diff.count() << " s"
         << "\n";
    std::cout << "Throughput: " << reps * matrices / diff.count() / 1000
         << "k matrices/s"
         << "\n";

    std::list<size_t> to_check;
    // We will check at least matrix 0
    to_check.push_back(0);
    // Spot check the last and the middle one
    if (matrices > 2) to_check.push_back(matrices / 2);
    if (matrices > 1) to_check.push_back(matrices - 1);


    // Count the number of errors found for this matrix
    int errorCount = 0;
    // Keep track of the max difference between the precomputed matrix using the
    // Gaussian elimination on the double datatype and the kernel computed 
    // inverse matrix using a QR based algorithm with the float datatype.
    double maxDiffBetweenSoftAndHard = 0.0;

    // For output post-processing (OP)
    TF invMatrixOP[kRows][kColumns];

    // Floating-point error threshold value at which we decide that the design
    // computed an incorrect value
    constexpr float kErrorThreshold = 1e-4;
    
    std::cout << "Verifying results on matrix ";
    for (size_t matrix : to_check) {
      std::cout << matrix << std::endl;

      // Read the inverse matrix from the output vector to invMatrixOP
      size_t idx = 0;
      for (size_t j = 0; j < kColumns; j++) {
        for (size_t i = 0; i < kRows; i++) {
          invMatrixOP[j][i] = invMatrix[matrix * kInverseMatrixSize + idx];
          idx++;
        }
      }

#ifdef DEBUG
      std::cout << "Kernel inverse" << std::endl;
      for(int row=0; row<kRows; row++){
        for(int col=0; col<kColumns; col++){
          std::cout << invMatrixOP[row][col] << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "Precomputed inverse" << std::endl;
      for(int row=0; row<kRows; row++){
        for(int col=0; col<kColumns; col++){
          std::cout << precomputedInvMatrix[matrix * kAMatrixSize + 
                                                    row*kColumns + col] << " ";
        }
        std::cout << std::endl;
      }
#endif

      // Keep track of the max difference between the precomputed inverse and
      // the kernel inverse
      double maxDiff = 0.0;

#if COMPLEX == 1
      for (size_t row = 0; row < kRows; row++) {
        for (size_t col = 0; col < kColumns; col++) {

          double diffR = abs(invMatrixOP[row][col].r() - 
          precomputedInvMatrix[matrix * kAMatrixSize + row*kColumns + col].r());

          double diffI = abs(invMatrixOP[row][col].i() - 
          precomputedInvMatrix[matrix * kAMatrixSize + row*kColumns + col].i());

          if(!std::isfinite(diffR) || !std::isfinite(diffR)){
            errorCount++;
          }

          if(diffR > maxDiff){
            maxDiff = diffR;
          }
          if(diffI > maxDiff){
            maxDiff = diffI;
          }

          if(diffR > kErrorThreshold){
            errorCount++;
          }
          if(diffI > kErrorThreshold){
            errorCount++;
          }
        }
      }
#else
      for (size_t i = 0; i < kRows; i++) {
        for (size_t j = 0; j < kColumns; j++) {

          double diff = abs(invMatrixOP[i][j] - 
                  precomputedInvMatrix[matrix * kAMatrixSize + i*kColumns + j]);

          if(!std::isfinite(diff)){
            errorCount++;
          }

          if(diff > maxDiff){
            maxDiff = diff;
          }

          if(diff > kErrorThreshold){
            errorCount++;
          }
        }
      }
#endif

      // Update the max diff 
      if(maxDiff > maxDiffBetweenSoftAndHard){
        maxDiffBetweenSoftAndHard = maxDiff;
      }

      // If an error was found, stop checking matrices
      if(errorCount>0){
        break;
      }
    } // end of matrix

    if (errorCount > 0) {
      std::cout << std::endl << "FAILED" << std::endl;
      std::cout << std::endl << "!!!!!!!!!!!!!! " << errorCount << " errors" 
                << std::endl;
      std::cout << "Max difference between the precomputed inverse and the "
                << "kernel value: " << maxDiffBetweenSoftAndHard << std::endl;
      return 1;
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
    std::cerr <<  "Caught a memory allocation exception on the host: " 
              << e.what() << std::endl;
    std::cerr <<  "   You can reduce the memory requirement by reducing the "
                  "number of matrices generated. Specify a smaller number when "
                  "running the executable."
              << std::endl;
    std::cerr << "   In this run, more than "
              << (((long long)matrices * (kAMatrixSize + kInverseMatrixSize) *
                  sizeof(float)) / pow(2, 30))
              << " GBs of memory was requested for " << matrices
              << " matrices, each of size " << kRows << " x "
              << kColumns << "\n";
    std::terminate();
  }
}