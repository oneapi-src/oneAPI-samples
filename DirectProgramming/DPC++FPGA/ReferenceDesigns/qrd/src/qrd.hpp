#ifndef __QRD_HPP__
#define __QRD_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_qrd.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class QRDDDRToLocalMem;
class QRD;
class QRDLocalMemToDDRQ;
class QRDLocalMemToDDRR;
class APipe;
class QPipe;
class RPipe;

/*
  Implementation of the QR decomposition using multiple streaming kernels
  Can be configured by datatype, matrix size and works with square or
  rectangular matrices, real and complex.
*/
template <unsigned columns,     // Number of columns in the input matrix
          unsigned rows,        // Number of rows in the input matrix
          unsigned raw_latency, // RAW latency for triangular loop optimization
          bool is_complex,      // Selects between ac_complex<T> and T datatype
          typename T,           // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
                        // TT will be ac_complex<T> or T depending on is_complex
         >
void QRDecompositionImpl(
  std::vector<TT> &a_matrix, // Input matrix to decompose
  std::vector<TT> &q_matrix, // Output matrix Q
  std::vector<TT> &r_matrix, // Output matrix R
  sycl::queue &q,            // Device queue
  int matrix_count,          // Number of matrices to decompose
  int repetitions           // Number of repetitions, for performance evaluation
) {

  constexpr int kAMatrixSize = columns * rows;
  constexpr int kQMatrixSize = columns * rows;
  constexpr int kRMatrixSize = columns * (columns + 1) / 2;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using QMatrixPipe = sycl::ext::intel::pipe<QPipe, PipeType, 3>;
  using RMatrixPipe = sycl::ext::intel::pipe<RPipe, TT,
                                                  kNumElementsPerDDRBurst * 4>;

  // Allocate FPGA DDR memory.
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *q_device = sycl::malloc_device<TT>(kQMatrixSize * matrix_count, q);
  TT *r_device = sycl::malloc_device<TT>(kRMatrixSize * matrix_count, q);

  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();

  // Launch the compute kernel and time the execution
  auto start_time = std::chrono::high_resolution_clock::now();

  q.submit([&](sycl::handler &h) {
    h.single_task<QRDDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                            AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the QR
  // decomposition. Write the Q and R output matrices to the QMatrixPipe
  // and RMatrixPipe pipes.
  q.single_task<QRD>(
      fpga_linalg::StreamingQRD<T, is_complex, rows, columns, raw_latency,
                   kNumElementsPerDDRBurst,
                   AMatrixPipe, QMatrixPipe, RMatrixPipe>());

  auto q_event = q.single_task<QRDLocalMemToDDRQ>([=
                                    ]() [[intel::kernel_args_restrict]] {
    // Read the Q matrix from the QMatrixPipe pipe and copy it to the
    // FPGA DDR
    MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                        QMatrixPipe>(q_device, matrix_count, repetitions);
  });

  auto r_event = q.single_task<QRDLocalMemToDDRR>([=
                                    ]() [[intel::kernel_args_restrict]] {
    // Read the R matrix from the RMatrixPipe pipe and copy it to the
    // FPGA DDR
    // Number of DDR burst of kNumElementsPerDDRBurst required to write
    // one vector
    constexpr bool kIncompleteBurst = kRMatrixSize%kNumElementsPerDDRBurst != 0;
    constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
    constexpr int kLoopIter = (kRMatrixSize / kNumElementsPerDDRBurst)
                              + kExtraIteration;

    sycl::device_ptr<TT> vector_ptr_device(r_device);


    // Repeat matrix_count complete R matrix pipe reads
    // for as many repetitions as needed
    for(int repetition_index = 0; repetition_index < repetitions;
                                                            repetition_index++){
      for(int matrix_index = 0; matrix_index < matrix_count; matrix_index++){

        [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
        [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
        TT r_result[kRMatrixSize/kNumElementsPerDDRBurst + kExtraIteration]
                                                      [kNumElementsPerDDRBurst];
        // Read a full R matrix
        for(int vector_elem = 0; vector_elem < kRMatrixSize; vector_elem++){
          r_result[vector_elem/kNumElementsPerDDRBurst]
                  [vector_elem%kNumElementsPerDDRBurst] = RMatrixPipe::read();

        } // end of vector_elem

        // Copy the R matrix result to DDR
        for (int li = 0; li < kLoopIter; li++) {
          if constexpr (kIncompleteBurst){
            // Write a burst of kNumElementsPerDDRBurst elements to DDR
            #pragma unroll
            for (int k = 0; k < kNumElementsPerDDRBurst; k++) {
              if(li * kNumElementsPerDDRBurst + k < kRMatrixSize){
                vector_ptr_device[matrix_index * kRMatrixSize
                          + li * kNumElementsPerDDRBurst + k] = r_result[li][k];
              }
            }
          }
          else{
            // Write a burst of kNumElementsPerDDRBurst elements to DDR
            #pragma unroll
            for (int k = 0; k < kNumElementsPerDDRBurst; k++) {
              vector_ptr_device[matrix_index * kRMatrixSize
                          + li * kNumElementsPerDDRBurst + k] = r_result[li][k];
            }
          }
        } // end of matrix_index
      }  // end of repetition_index
    }  // end of li
  });

  q_event.wait();
  r_event.wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff.count() << " s" << std::endl;
  std::cout << "Throughput: "
            << repetitions * matrix_count / diff.count() * 1e-3
            << "k matrices/s" << std::endl;


  // Copy the Q and R matrices result from the FPGA DDR to the host memory
  q.memcpy(q_matrix.data(), q_device, kQMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();
  q.memcpy(r_matrix.data(), r_device, kRMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(q_device, q);
  free(r_device, q);
}

#endif /* __QRD_HPP__ */