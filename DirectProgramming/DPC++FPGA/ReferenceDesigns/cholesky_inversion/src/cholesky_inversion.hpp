#ifndef __CHOLESKY_INVERSION_HPP__
#define __CHOLESKY_INVERSION_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"

// Included from DirectProgramming/DPC++FPGA/include/
#include "streaming_cholesky.hpp"
#include "streaming_cholesky_inversion.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class CholeskyDDRToLocalMem;
class CholeskyDecomposition;
class CholeskyInversion;
class CholeskyLocalMemToDDRL;
class APipe;
class LPipe;
class IPipe;

/*
  Implementation of the Cholesky-based inversion using streaming kernels
  Can be configured by datatype, matrix size (must use square matrices), real
  and complex.
*/
template <unsigned dimension,  // Number of columns/rows in the input matrix
          unsigned raw_latency_decomposition,  // RAW latency for triangular
                                               // loop optimization
          unsigned raw_latency_inversion,  // RAW latency for triangular loop
                                           // optimization
          bool is_complex,  // Selects between ac_complex<T> and T datatype
          typename T,       // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
          // TT will be ac_complex<T> or T depending on is_complex
          >
void CholeskyInversionImpl(
    std::vector<TT> &a_matrix,  // Input matrix A to invert
    std::vector<TT> &i_matrix,  // Output matrix I (inverse of A)
    sycl::queue &q,             // Device queue
    int matrix_count,           // Number of matrices to invert
    int repetitions             // Number of repetitions, for performance
                                // evaluation
) {
  constexpr int kAMatrixSize = dimension * dimension;
  constexpr int kIMatrixSize = dimension * (dimension + 1) / 2;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A and L matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using LMatrixPipe =
      sycl::ext::intel::pipe<LPipe, TT, kNumElementsPerDDRBurst * 4>;
  // using IMatrixPipe = sycl::ext::intel::pipe<IPipe, PipeType, 3>;
  using IMatrixPipe = sycl::ext::intel::pipe<IPipe, TT, kNumElementsPerDDRBurst * 4>;

  // Allocate FPGA DDR memory.
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *i_device = sycl::malloc_device<TT>(kIMatrixSize * matrix_count, q);

  if ((a_device == nullptr) || (i_device == nullptr)) {
    std::cerr << "Error when allocating FPGA DDR" << std::endl;
    std::cerr << "The FPGA DDR may be full" << std::endl;
    std::cerr << "Try reducing the matrix sizes/count" << std::endl;
    return;
  }

  // Copy the matrices to decompose on the FPGA DDR
  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Launch a kernel that will repeatedly read the matrices on the FPGA DDR
  // and write their content to the AMatrixPipe pipe.
  auto ddr_read_event = q.single_task<CholeskyDDRToLocalMem>([=] {
    MatrixReadFromDDRToPipe<TT, dimension, dimension, kNumElementsPerDDRBurst,
                            AMatrixPipe>(a_device, matrix_count, repetitions);
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the Cholesky
  // decomposition. Write the L output matrix to the LMatrixPipe pipe.
  q.single_task<CholeskyDecomposition>(
      fpga_linalg::StreamingCholesky<
          T, is_complex, dimension, raw_latency_decomposition,
          kNumElementsPerDDRBurst, AMatrixPipe, LMatrixPipe>());

  // Read the L matrix from the LMatrixPipe pipe and compute the Cholesky-based
  // inversion. Write the I output matrix to the IMatrixPipe pipe.
  q.single_task<CholeskyInversion>(
      fpga_linalg::StreamingCholeskyInversion<
          T, is_complex, dimension, raw_latency_inversion,
          kNumElementsPerDDRBurst, LMatrixPipe, IMatrixPipe>());

  // auto ddr_write_event = q.single_task<CholeskyLocalMemToDDRL>([=] {
  //   MatrixReadFromPipeToDDR<TT, dimension, dimension, kNumElementsPerDDRBurst,
  //                           IMatrixPipe>(i_device, matrix_count, repetitions);
  // });


  auto ddr_write_event = q.single_task<CholeskyLocalMemToDDRL>([=
                                    ]() [[intel::kernel_args_restrict]] {
    // Read the R matrix from the RMatrixPipe pipe and copy it to the
    // FPGA DDR
    // Number of DDR burst of kNumElementsPerDDRBurst required to write
    // one vector
    constexpr int kRMatrixSize = dimension * (dimension + 1) / 2;
    constexpr bool kIncompleteBurst = kRMatrixSize%kNumElementsPerDDRBurst != 0;
    constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
    constexpr int kLoopIter = (kRMatrixSize / kNumElementsPerDDRBurst)
                              + kExtraIteration;

    sycl::device_ptr<TT> vector_ptr_device(i_device);

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
                  [vector_elem%kNumElementsPerDDRBurst] = IMatrixPipe::read();

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



  ddr_write_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_read_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: " << ((repetitions * matrix_count) / diff) * 1e-3
            << "k matrices/s" << std::endl;

  // Copy the L matrices result from the FPGA DDR to the host memory
  q.memcpy(i_matrix.data(), i_device, kIMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(i_device, q);
}

#endif /* __CHOLESKY_INVERSION_HPP__ */
