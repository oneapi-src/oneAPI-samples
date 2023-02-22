#ifndef __QRD_HPP__
#define __QRD_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include <chrono>
#include <cstring>
#include <type_traits>
#include <vector>

#include "memory_transfers.hpp"
#include "streaming_eigen.hpp"
#include "CovMatrix.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class QRDDDRToLocalMem;
class COV;
class EIGEN;
class QRDLocalMemToDDRQ;
class QRDLocalMemToDDRR;
class CPipe;
class APipe;
class CMatrixPipe;
class RQPipe;
class QQPipe;

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
  std::vector<TT> &eig_matrix, // Output matrix Q
  std::vector<TT> &qq_matrix, // Output matrix R
  sycl::queue &q,            // Device queue
  int matrix_count,          // Number of matrices to decompose
  int repetitions           // Number of repetitions, for performance evaluation
) {

  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;
  constexpr int kAMatrixSize = SAMPE_SIZE * rows;
  constexpr int kQQMatrixSize = columns * rows;
  constexpr int kEigMatrixSize = rows + 1; // additional one for debug data
  

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  // Pipes to communicate the A, Q and R matrices between kernels
  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using CMatrixPipe = sycl::ext::intel::pipe<CPipe, PipeType, 3>;
  using EigMatrixPipe = sycl::ext::intel::pipe<RQPipe, PipeType, 3>;
  using QQMatrixPipe = sycl::ext::intel::pipe<QQPipe, PipeType, 3>;


  // Allocate FPGA DDR memory.
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *qq_device = sycl::malloc_device<TT>(kQQMatrixSize * matrix_count, q);
  TT *eig_device = sycl::malloc_device<TT>(kEigMatrixSize * matrix_count, q);

  q.memcpy(a_device, a_matrix.data(), kAMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();

  auto ddr_write_event =
  q.submit([&](sycl::handler &h) {
    h.single_task<QRDDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, SAMPE_SIZE, rows, kNumElementsPerDDRBurst,
                            AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });


  q.single_task<COV>(
    fpga_linalg::StreamingMM<T, is_complex, rows, SAMPE_SIZE, kNumElementsPerDDRBurst,
            kNumElementsPerDDRBurst, AMatrixPipe, CMatrixPipe>());


  // Read the A matrix from the AMatrixPipe pipe and compute the QR
  // decomposition. Write the Q and R output matrices to the QMatrixPipe
  // and RMatrixPipe pipes.

  q.single_task<EIGEN>(
    fpga_linalg::StreamingQRD<T, is_complex, rows, columns, raw_latency,
                kNumElementsPerDDRBurst, CMatrixPipe, EigMatrixPipe, QQMatrixPipe>());


  auto eig_event = q.single_task<QRDLocalMemToDDRR>([=
                                    ]() [[intel::kernel_args_restrict]] {
    // Read the R matrix from the RMatrixPipe pipe and copy it to the
    // FPGA DDR
    sycl::device_ptr<TT> vector_ptr_device(eig_device);

    // Repeat matrix_count complete R matrix pipe reads
    // for as many repetitions as needed
     MatrixReadPipeToDDR<TT, rows+1, 1, kNumElementsPerDDRBurst,
                        EigMatrixPipe>(eig_device, matrix_count, repetitions);
  });


  auto qq_event = q.single_task<QRDLocalMemToDDRQ>([=
                                    ]() [[intel::kernel_args_restrict]] {
    // Read the Q matrix from the QMatrixPipe pipe and copy it to the
    // FPGA DDR
    sycl::device_ptr<TT> vector_ptr_device(qq_device);
    MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
                        QQMatrixPipe>(qq_device, matrix_count, repetitions);
  });



  qq_event.wait();
  eig_event.wait();

  // Compute the total time the execution lasted
  auto start_time = ddr_write_event.template
              get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time = qq_event.template
                get_profiling_info<sycl::info::event_profiling::command_end>();
  double diff = (end_time - start_time) / 1.0e9;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff << " s" << std::endl;
  std::cout << "Throughput: "
            << repetitions * matrix_count / diff * 1e-3
            << "k matrices/s" << std::endl;


  // Copy the Q and R matrices result from the FPGA DDR to the host memory
  q.memcpy(qq_matrix.data(), qq_device, kQQMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();
  q.memcpy(eig_matrix.data(), eig_device, kEigMatrixSize * matrix_count
                                                          * sizeof(TT)).wait();

  // Clean allocated FPGA memory
  free(a_device, q);
  free(eig_device, q);
  free(qq_device, q);
}

#endif /* __QRD_HPP__ */
