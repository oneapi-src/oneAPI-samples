#pragma once

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
#include "streaming_qri.hpp"
#include "tuple.hpp"

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class QRIDDRToLocalMem;
class QRD;
class QRI;
class QRILocalMemToDDRQ;
class APipe;
class QPipe;
class RPipe;
class IPipe;

template <unsigned columns,         // Number of columns in the input matrix
          unsigned rows,            // Number of rows in the input matrix
          unsigned raw_latency_qrd, // RAW latency for triangular loop
                                    // optimization in the QRD kernel
          unsigned raw_latency_qri, // RAW latency for triangular loop
                                    // optimization in the QRI kernel
          bool is_complex,          // Selects between ac_complex<T> and T
                                    // datatype
          typename T,               // The datatype for the computation
          typename TT = std::conditional_t<is_complex, ac_complex<T>, T>
                                    // TT will be ac_complex<T> or T depending
                                    // on is_complex
          >
void QRIImpl(
    std::vector<TT> &a_matrix,       // Input matrix to inverse
    std::vector<TT> &inverse_matrix, // Output inverse matrix
    sycl::queue &q,                  // Device queue
    size_t matrix_count,             // Number of matrices to process
    size_t repetitions               // Number of repetitions (for performance
                                     // evaluation)
) {

  // Functional limitations
  static_assert(
      rows >= columns,
      "only rectangular matrices with rows>=columns are matrices supported");
  static_assert(columns >= 4,
                "only matrices of size 4x4 or over are supported");

  constexpr int kAMatrixSize = rows * columns;
  constexpr int kInverseMatrixSize = rows * columns;
  constexpr int kNumElementsPerDDRBurst = is_complex ? 4 : 8;

  using PipeType = fpga_tools::NTuple<TT, kNumElementsPerDDRBurst>;

  using AMatrixPipe = sycl::ext::intel::pipe<APipe, PipeType, 3>;
  using QMatrixPipe = sycl::ext::intel::pipe<QPipe, PipeType, 3>;
  using RMatrixPipe = sycl::ext::intel::pipe<RPipe, TT, 3>;
  using InverseMatrixPipe = sycl::ext::intel::pipe<IPipe, PipeType, 3>;


  // Create buffers and allocate space for them.
  TT *a_device = sycl::malloc_device<TT>(kAMatrixSize * matrix_count, q);
  TT *i_device = sycl::malloc_device<TT>(kInverseMatrixSize * matrix_count, q);

  q.memcpy(a_device, a_matrix.data(),
                             kAMatrixSize * matrix_count * sizeof(TT)).wait();

  // Launch the compute kernel and time the execution
  auto start_time = std::chrono::high_resolution_clock::now();

  q.submit([&](sycl::handler &h) {
    h.single_task<QRIDDRToLocalMem>([=]() [[intel::kernel_args_restrict]] {
      MatrixReadFromDDRToPipe<TT, rows, columns, kNumElementsPerDDRBurst,
                            AMatrixPipe>(a_device, matrix_count, repetitions);
    });
  });

  // Read the A matrix from the AMatrixPipe pipe and compute the QR
  // decomposition. Write the Q and R output matrices to the QMatrixPipe
  // and RMatrixPipe pipes.
  q.single_task<QRD>(
      fpga_linalg::StreamingQRD<T, is_complex, rows, columns, raw_latency_qrd,
                   kNumElementsPerDDRBurst,
                   AMatrixPipe, QMatrixPipe, RMatrixPipe>());

  q.single_task<QRI>(
      // Read the Q and R matrices from pipes and compute the inverse of A.
      // Write the result to the InverseMatrixPipe pipe.
      fpga_linalg::StreamingQRI<T, is_complex, rows, columns, raw_latency_qri,
                   kNumElementsPerDDRBurst,
                   QMatrixPipe, RMatrixPipe, InverseMatrixPipe>());


  auto i_event = q.single_task<QRILocalMemToDDRQ>([=
                                      ]() [[intel::kernel_args_restrict]] {
      // Read the inverse matrix from the InverseMatrixPipe pipe and copy it
      // to the FPGA DDR
      MatrixReadPipeToDDR<TT, rows, columns, kNumElementsPerDDRBurst,
              InverseMatrixPipe>(i_device, matrix_count, repetitions);
  });

  i_event.wait();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  q.throw_asynchronous();

  std::cout << "   Total duration:   " << diff.count() << " s" << std::endl;
  std::cout << "Throughput: "
            << repetitions * matrix_count / diff.count() * 1e-3
            << "k matrices/s" << std::endl;


  // Copy the Q and R matrices result from the FPGA DDR to the host memory
  q.memcpy(inverse_matrix.data(), i_device,
               kInverseMatrixSize * matrix_count * sizeof(TT)).wait();

  // Clean allocated FPGA memory
    free(a_device, q);
    free(i_device, q);
}