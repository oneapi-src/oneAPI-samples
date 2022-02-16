#ifndef __CHOLESKY_HPP__
#define __CHOLESKY_HPP__

#include <CL/sycl.hpp>
#include <chrono>
#include <cstring>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_cholesky.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class CholeskyDDRToLocalMem;
class Cholesky;
class CholeskyLocalMemToDDRL;
class APipe;
class QPipe;
class RPipe;

/*
  Implementation of the Cholesky decomposition using multiple streaming kernels
  Can be configured by datatype, matrix size (must use square matrices), real
  and complex.
*/
template <unsigned columns,      // Number of columns in the input matrix
          unsigned rows,         // Number of rows in the input matrix
          unsigned raw_latency,  // RAW latency for triangular loop optimization
          bool is_complex,       // Selects between ac_complex<T> and T datatype
          typename T,            // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
          // TT will be ac_complex<T> or T depending on is_complex
          >
void CholeskyDecompositionImpl(
    std::vector<TT> &a_matrix,  // Input matrix to decompose
    std::vector<TT> &l_matrix,  // Output matrix R
    sycl::queue &q,             // Device queue
    int matrix_count,           // Number of matrices to decompose
    int repetitions  // Number of repetitions, for performance evaluation
) {
  constexpr int kAMatrixSize = columns * rows;
  constexpr int kLMatrixSize = columns * (columns + 1) / 2;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A and L matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using LMatrixPipe =
      sycl::ext::intel::pipe<RPipe, TT, kNumElementsPerDDRBurst * 4>;

  // Allocate FPGA DDR memory.
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *l_device = sycl::malloc_device<TT>(kLMatrixSize * matrix_count, q);

  // Copy the matrices to decompose on the FPGA DDR
  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Launch the compute kernel and time the execution
  auto start_time = std::chrono::high_resolution_clock::now();

  // Launch a kernel that will repeatedly read the matrices on the FPGA DDR
  // and write their content to the AMatrixPipe pipe.
  q.submit([&](sycl::handler &h) {
    h.single_task<CholeskyDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                              AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the Cholesky
  // decomposition. Write the L output matrix to the LMatrixPipe pipe.
  q.single_task<Cholesky>(
      StreamingCholesky<T, is_complex, rows, raw_latency,
                        kNumElementsPerDDRBurst, AMatrixPipe, LMatrixPipe>());

  q.single_task<CholeskyLocalMemToDDRL>([=]() [[intel::kernel_args_restrict]] {
     // Read the L matrix from the LMatrixPipe pipe and copy it to the
     // FPGA DDR
     // Number of DDR burst of kNumElementsPerDDRBurst required to write
     // one vector
     constexpr bool kIncompleteBurst =
         kLMatrixSize % kNumElementsPerDDRBurst != 0;
     constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
     constexpr int kLoopIter =
         (kLMatrixSize / kNumElementsPerDDRBurst) + kExtraIteration;

     sycl::device_ptr<TT> vector_ptr_device(l_device);

     // Repeat matrix_count complete L matrix pipe reads
     // for as many repetitions as needed
     for (int repetition_index = 0; repetition_index < repetitions;
          repetition_index++) {
       for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
         [[intel::private_copies(4)]]  // NO-FORMAT: Attribute
         [[intel::max_replicates(1)]]  // NO-FORMAT: Attribute
         TT l_result[kLMatrixSize / kNumElementsPerDDRBurst + kExtraIteration]
                    [kNumElementsPerDDRBurst];
         // Read a full L matrix
         for (int vector_elem = 0; vector_elem < kLMatrixSize; vector_elem++) {
           l_result[vector_elem / kNumElementsPerDDRBurst]
                   [vector_elem % kNumElementsPerDDRBurst] =
                       LMatrixPipe::read();

         }  // end of vector_elem

         // Copy the L matrix result to DDR
         for (int li = 0; li < kLoopIter; li++) {
           if constexpr (kIncompleteBurst) {
// Write a burst of kNumElementsPerDDRBurst elements to DDR
#pragma unroll
             for (int k = 0; k < kNumElementsPerDDRBurst; k++) {
               if (li * kNumElementsPerDDRBurst + k < kLMatrixSize) {
                 vector_ptr_device[matrix_index * kLMatrixSize +
                                   li * kNumElementsPerDDRBurst + k] =
                     l_result[li][k];
               }
             }
           } else {
// Write a burst of kNumElementsPerDDRBurst elements to DDR
#pragma unroll
             for (int k = 0; k < kNumElementsPerDDRBurst; k++) {
               vector_ptr_device[matrix_index * kLMatrixSize +
                                 li * kNumElementsPerDDRBurst + k] =
                   l_result[li][k];
             }
           }
         }  // end of matrix_index
       }    // end of repetition_index
     }      // end of li
   }).wait();

  // End the timer as all the matrices have been processed.
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff.count() << " s" << std::endl;
  std::cout << "Throughput: "
            << repetitions * matrix_count / diff.count() * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the L matrices result from the FPGA DDR to the host memory
  q.memcpy(l_matrix.data(), l_device, kLMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(l_device, q);
}

#endif /* __CHOLESKY_HPP__ */