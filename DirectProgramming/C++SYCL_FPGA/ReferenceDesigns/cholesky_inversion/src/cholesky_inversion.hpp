#ifndef __CHOLESKY_INVERSION_HPP__
#define __CHOLESKY_INVERSION_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"

// Included from DirectProgramming/C++SYCL_FPGA/include/
#include "streaming_cholesky.hpp"
#include "streaming_cholesky_inversion.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class CholeskyDDRToLocalMem;
class DecompositionKernel;
class InversionKernel;
class CholeskyLocalMemToDDR;
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
  using IMatrixPipe =
      sycl::ext::intel::pipe<IPipe, TT, kNumElementsPerDDRBurst * 4>;

  // Allocate FPGA DDR memory.
#if defined (IS_BSP)
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *i_device = sycl::malloc_device<TT>(kIMatrixSize * matrix_count, q);
#else
  // malloc_device are not supported when targetting an FPGA part/family
  TT *a_device = sycl::malloc_shared<TT>(kAMatrixSize * matrix_count, q);
  TT *i_device = sycl::malloc_shared<TT>(kIMatrixSize * matrix_count, q);
#endif  

  if ((a_device == nullptr) || (i_device == nullptr)) {
    std::cerr << "Error when allocating FPGA DDR" << std::endl;
    std::cerr << "The FPGA DDR may be full" << std::endl;
    std::cerr << "Try reducing the matrix sizes/count" << std::endl;
    return;
  }

  // Copy the matrices to decompose to the FPGA DDR
  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count * sizeof(TT))
      .wait();

  // Launch a kernel that will repeatedly read the matrices from the FPGA DDR
  // and write their content to the AMatrixPipe pipe.
  auto ddr_read_event = q.single_task<CholeskyDDRToLocalMem>([=] {
    MatrixReadFromDDRToPipe<TT, dimension, dimension, kNumElementsPerDDRBurst,
                            AMatrixPipe>(a_device, matrix_count, repetitions);
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the Cholesky
  // decomposition. Write the L output matrix to the LMatrixPipe pipe.
  q.single_task<DecompositionKernel>(
      fpga_linalg::StreamingCholesky<
          T, is_complex, dimension, raw_latency_decomposition,
          kNumElementsPerDDRBurst, AMatrixPipe, LMatrixPipe>());

  // Read the L matrix from the LMatrixPipe pipe and compute the Cholesky-based
  // inversion. Write the I output matrix to the IMatrixPipe pipe.
  q.single_task<InversionKernel>(
      fpga_linalg::StreamingCholeskyInversion<
          T, is_complex, dimension, raw_latency_inversion,
          kNumElementsPerDDRBurst, LMatrixPipe, IMatrixPipe>());

  // Read the I matrix from the LMatrixPipe pipe and copy it to the FPGA DDR
  auto ddr_write_event = q.single_task<CholeskyLocalMemToDDR>([=] {
    VectorReadFromPipeToDDR<TT, kIMatrixSize, kNumElementsPerDDRBurst,
                            IMatrixPipe>(i_device, matrix_count, repetitions);
  });

  ddr_write_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_read_event.template get_profiling_info<
      sycl::info::event_profiling::command_start>();
  auto end_time = ddr_write_event.template get_profiling_info<
      sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;

  // Make sure we throw any asynchronous errors if they have occurred during 
  // the computation
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
