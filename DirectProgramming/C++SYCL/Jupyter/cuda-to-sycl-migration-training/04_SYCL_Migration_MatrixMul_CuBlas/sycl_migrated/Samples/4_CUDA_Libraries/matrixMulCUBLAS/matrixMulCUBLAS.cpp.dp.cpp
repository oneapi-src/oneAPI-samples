/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
* Matrix multiplication: C = A * B.
* Host code.
*
* This sample implements matrix multiplication as described in Chapter 3
* of the programming guide and uses the CUBLAS library to demonstrate
* the best performance.

* SOME PRECAUTIONS:
* IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
* WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
* The reason is explained as follows:

* CUBLAS library uses column-major storage, but C/C++ use row-major storage.
* When passing the matrix pointer to CUBLAS, the memory layout alters from
* row-major to column-major, which is equivalent to an implicit transpose.

* In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
* C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
* implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
* If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
* multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
* is a column-based cublas matrix, which means C(T) in C/C++, we need extra
* transpose code to convert it to a row-based C/C++ matrix.

* To solve the problem, let's consider our desired result C, a row-major matrix.
* In cublas format, it is C(T) actually (because of the implicit transpose).
* C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
* happen to be C/C++ matrice B and A (still because of the implicit transpose)!
* We don't need extra transpose code, we only need alter the input order!
*
* CUBLAS provides high-performance matrix multiplication.
* See also:
* V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
* in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
* Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
*/

// Utilities and system includes
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <helper_string.h>
#include <dpct/blas_utils.hpp>
  // helper for shared functions common to CUDA Samples

// CUDA runtime

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

#ifndef min
#define min(a, b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a, b) ((a > b) ? a : b)
#endif

// Optional Command-line multiplier for matrix sizes
typedef struct _matrixSize {
  unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA,
                  unsigned int wA, unsigned int wB) {
  for (unsigned int i = 0; i < hA; ++i)
    for (unsigned int j = 0; j < wB; ++j) {
      double sum = 0;

      for (unsigned int k = 0; k < wA; ++k) {
        double a = A[i * wA + k];
        double b = B[k * wB + j];
        sum += a * b;
      }

      C[i * wB + j] = (float)sum;
    }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size) {
  for (int i = 0; i < size; ++i) data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height,
               int iListLength, float fListTol) {
  printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
  int i, j, k;
  int error_count = 0;

  for (j = 0; j < height; j++) {
    if (error_count < iListLength) {
      printf("\n  Row %d:\n", j);
    }

    for (i = 0; i < width; i++) {
      k = j * width + i;
      float fDiff = fabs(data1[k] - data2[k]);

      if (fDiff > fListTol) {
        if (error_count < iListLength) {
          printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j,
                 data1[k], data2[k], fDiff);
        }

        error_count++;
      }
    }
  }

  printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple,
                    sMatrixSize &matrix_size) try {
  // By default, we use device 0, otherwise we override the device ID based on
  // what is provided at the command line
  dpct::err0 error;
  devID = 0;

  devID = findCudaDevice(argc, (const char **)argv);

  if (checkCmdLineFlag(argc, (const char **)argv, "sizemult")) {
    iSizeMultiple =
        getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
  }

  iSizeMultiple = min(iSizeMultiple, 10);
  iSizeMultiple = max(iSizeMultiple, 1);

  dpct::device_info deviceProp;

  error = DPCT_CHECK_ERROR(dpct::get_device(devID).get_device_info(deviceProp));

  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID,
         /*
         DPCT1005:14: The SYCL device version is different from CUDA Compute
         Compatibility. You may need to rewrite this code.
         */
         deviceProp.get_name(), deviceProp.get_major_version(),
         deviceProp.get_minor_version());

  int block_size = 32;

  matrix_size.uiWA = 3 * block_size * iSizeMultiple;
  matrix_size.uiHA = 4 * block_size * iSizeMultiple;
  matrix_size.uiWB = 2 * block_size * iSizeMultiple;
  matrix_size.uiHB = 3 * block_size * iSizeMultiple;
  matrix_size.uiWC = 2 * block_size * iSizeMultiple;
  matrix_size.uiHC = 4 * block_size * iSizeMultiple;

  printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n", matrix_size.uiHA,
         matrix_size.uiWA, matrix_size.uiHB, matrix_size.uiWB, matrix_size.uiHC,
         matrix_size.uiWC);

  if (matrix_size.uiWA != matrix_size.uiHB ||
      matrix_size.uiHA != matrix_size.uiHC ||
      matrix_size.uiWB != matrix_size.uiWC) {
    printf("ERROR: Matrix sizes do not match!\n");
    exit(-1);
  }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size) {
  dpct::device_info deviceProp;

  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::get_device(devID).get_device_info(deviceProp)));

  int block_size = 32;

  // set seed for rand()
  srand(2006);

  // allocate host memory for matrices A and B
  unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A = (float *)malloc(mem_size_A);
  unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B = (float *)malloc(mem_size_B);

  // set seed for rand()
  srand(2006);

  // initialize host memory
  randomInit(h_A, size_A);
  randomInit(h_B, size_B);

  // allocate device memory
  float *d_A, *d_B, *d_C;
  unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
  unsigned int mem_size_C = sizeof(float) * size_C;

  // allocate host memory for the result
  float *h_C = (float *)malloc(mem_size_C);
  float *h_CUBLAS = (float *)malloc(mem_size_C);

  checkCudaErrors(
      DPCT_CHECK_ERROR(d_A = (float *)sycl::malloc_device(
                           mem_size_A, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_B = (float *)sycl::malloc_device(
                           mem_size_B, dpct::get_in_order_queue())));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_A, h_A, mem_size_A).wait()));
  checkCudaErrors(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d_B, h_B, mem_size_B).wait()));
  checkCudaErrors(
      DPCT_CHECK_ERROR(d_C = (float *)sycl::malloc_device(
                           mem_size_C, dpct::get_in_order_queue())));

  // setup execution parameters
  dpct::dim3 threads(block_size, block_size);
  dpct::dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

  // create and start timer
  printf("Computing result using CUBLAS...");

  // execute the kernel
  int nIter = 30;

  // CUBLAS version 2.0
  {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    dpct::blas::descriptor_ptr handle;
    dpct::event_ptr start, stop;

    checkCudaErrors(DPCT_CHECK_ERROR(handle = new dpct::blas::descriptor()));

    // Perform warmup operation with cublas
    checkCudaErrors(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        handle->get_queue(), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, matrix_size.uiWB, matrix_size.uiHA,
        matrix_size.uiWA, alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA,
        beta, d_C, matrix_size.uiWB)));

    // Allocate CUDA events that we'll use for timing
    checkCudaErrors(DPCT_CHECK_ERROR(start = new sycl::event()));
    checkCudaErrors(DPCT_CHECK_ERROR(stop = new sycl::event()));

    // Record the start event
    /*
    DPCT1024:15: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(start, &dpct::get_in_order_queue())));

    for (int j = 0; j < nIter; j++) {
      // note cublas is column primary!
      // need to transpose the order
      checkCudaErrors(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
          handle->get_queue(), oneapi::mkl::transpose::nontrans,
          oneapi::mkl::transpose::nontrans, matrix_size.uiWB, matrix_size.uiHA,
          matrix_size.uiWA, alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA,
          beta, d_C, matrix_size.uiWB)));
    }

    printf("done.\n");

    // Record the stop event
    /*
    DPCT1024:16: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::sync_barrier(stop, &dpct::get_in_order_queue())));

    // Wait for the stop event to complete
    checkCudaErrors(DPCT_CHECK_ERROR(stop->wait_and_throw()));

    float msecTotal = 0.0f;
    checkCudaErrors(DPCT_CHECK_ERROR(
        msecTotal = (stop->get_profiling_info<
                         sycl::info::event_profiling::command_end>() -
                     start->get_profiling_info<
                         sycl::info::event_profiling::command_start>()) /
                    1000000.0f));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC *
                               (double)matrix_size.uiWC *
                               (double)matrix_size.uiHB;
    double gigaFlops =
        (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    // copy result from device to host
    checkCudaErrors(DPCT_CHECK_ERROR(
        dpct::get_in_order_queue().memcpy(h_CUBLAS, d_C, mem_size_C).wait()));

    // Destroy the handle
    checkCudaErrors(DPCT_CHECK_ERROR(delete (handle)));
  }

  // compute reference solution
  printf("Computing result using host CPU...");
  float *reference = (float *)malloc(mem_size_C);
  matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA,
               matrix_size.uiWB);
  printf("done.\n");

  // check result (CUBLAS)
  bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);

  if (resCUBLAS != true) {
    printDiff(reference, h_CUBLAS, matrix_size.uiWC, matrix_size.uiHC, 100,
              1.0e-5f);
  }

  printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n",
         (true == resCUBLAS) ? "PASS" : "FAIL");

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n");

  // clean up memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(reference);
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())));
  checkCudaErrors(
      DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue())));

  if (resCUBLAS == true) {
    return EXIT_SUCCESS;  // return value = 1
  } else {
    return EXIT_FAILURE;  // return value = 0
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf("[Matrix Multiply CUBLAS] - Starting...\n");

  int devID = 0, sizeMult = 5;
  sMatrixSize matrix_size;

  initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

  int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

  return matrix_result;
}
