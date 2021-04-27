#ifndef __TRANSPOSE_HPP__
#define __TRANSPOSE_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "Tuple.hpp"
#include "UnrolledLoop.hpp"

using namespace sycl;

// the generic transpose class. Defined below after SubmitTransposeKernel.
template <typename T, size_t k_num_cols_in, size_t k_pipe_width,
          typename MatrixInPipe, typename MatrixOutPipe>
struct Transposer;

// SubmitTransposeKernel
// Accept k_pipe_width elements of type T wrapped in NTuple
// on each read from DataInPipe.  Store elements locally until we can write
// to the output array of pipes with data in transposed order.
// For simplicity of terminology, incoming data is defined to enter in 'row'
// order and exit in 'column' order, although for the purposes of the transpose
// the terms 'row' and 'column' could be interchanged.
// Row order means all data from a given row is received before any data from
// the next row is received.
// This kernel also performs flow control and ensures that no partial matrices
// will be written into the output pipe.
template <typename TransposeKernelName,  // Name to use for the Kernel
          typename T,                    // type of element to transpose
          size_t k_num_cols_in,   // number of columns in the input matrix
          size_t k_pipe_width,    // number of elements read/written
                                  // (wrapped in NTuple) from/to pipes
          typename MatrixInPipe,  // Receive the input matrix in row order
                                  // Receive k_pipe_width elements of type
                                  // T wrapped in NTuple on each read
          typename MatrixOutPipe  // Send the output matrix in column order.
                                  // Send k_pipe_width elements of type T
                                  // wrapped in NTuple on each write.
          >
event SubmitTransposeKernel(queue& q) {
  // Template parameter checking
  static_assert(std::numeric_limits<short>::max() > k_num_cols_in,
                "k_num_cols_in must fit in a short");
  static_assert(k_num_cols_in % k_pipe_width == 0,
                "k_num_cols_in must be evenly divisible by k_pipe_width");

  return q.submit([&](handler& h) {
    h.single_task<TransposeKernelName>([=]() {
      // start the transposer
      Transposer<T, k_num_cols_in, k_pipe_width, MatrixInPipe, MatrixOutPipe>
          TheTransposer;
      TheTransposer();
    });
  });
}

// The generic transpose. We use classes here because we want to do partial
// template specialization on the 'k_pipe_width' to optimize the case where
// 'k_pipe_width' == 1 and this cannot be done with a function.
template <typename T, size_t k_num_cols_in, size_t k_pipe_width,
          typename MatrixInPipe, typename MatrixOutPipe>
struct Transposer {
  void operator()() const {
    using PipeType = NTuple<T, k_pipe_width>;

    // This is a scratch pad memory that we will use to do the transpose.
    // We read the data in from a pipe (k_pipe_width elements at at time),
    // store it in this memory in row-major format and read it out in
    // column-major format (again, k_pipe_width elements at a time).
    constexpr int kNumScratchMemCopies = 4;   // must be a power of 2
    static_assert(kNumScratchMemCopies > 0);
    static_assert((kNumScratchMemCopies & (kNumScratchMemCopies - 1)) == 0);
    constexpr unsigned char kNumScratchMemCopiesBitMask = 
      kNumScratchMemCopies - 1;
    constexpr int kBankwidth = k_pipe_width * sizeof(T);
    // NO-FORMAT comments are for clang-format
    [[intel::numbanks(1)]]                   // NO-FORMAT: Attribute
    [[intel::bankwidth(kBankwidth)]]         // NO-FORMAT: Attribute
    [[intel::private_copies(1)]]             // NO-FORMAT: Attribute
    [[intel::max_replicates(k_pipe_width)]]  // NO-FORMAT: Attribute
    T scratch[kNumScratchMemCopies][k_pipe_width][k_num_cols_in];

    // track the status of each of the buffers
    // NO-FORMAT comments are for clang-format
    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    bool ready_to_send[kNumScratchMemCopies] = {false, false, false, false};

    unsigned char rx_buffer = 0;
    unsigned short rx_count = 0;
    unsigned char tx_buffer = 0;
    unsigned char tx_col = 0;
    bool last_tx_col = false;

    // create a 'pipeline' for the almost full signal
    constexpr int kAlmostFullPipeDepth = 2;
    NTuple<bool, kAlmostFullPipeDepth> almost_full_pipeline;
    UnrolledLoop<kAlmostFullPipeDepth>([&](auto pipe_stage) {
      almost_full_pipeline.template get<pipe_stage>() = false;
    });

    // NO-FORMAT comments are for clang-format
    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (1) {
      // capture current value of all status variables as we begin each loop
      // iteration
      unsigned char cur_rx_buffer = rx_buffer;
      unsigned short cur_rx_count = rx_count;
      unsigned char cur_tx_buffer = tx_buffer;
      unsigned char cur_tx_col = tx_col;
      bool cur_last_tx_col = last_tx_col;

      // Create a 'pipeline' for the almost full signal.
      // Calculate almost full at the end of the pipeline, and on each loop
      // iteration we shift the data down the pipeline.  Since we are using
      // an 'almost' full signal, we don't need the result right away, we
      // can wait several loop iterations.  This allows us to break
      // dependencies between loop iterations and improve FMAX.
      UnrolledLoop<kAlmostFullPipeDepth - 1>([&](auto pipe_stage) {
        almost_full_pipeline.template get<pipe_stage>() =
            almost_full_pipeline.template get<pipe_stage + 1>();
      });
      bool cur_almost_full = almost_full_pipeline.first();

      // Calculate almost full at the start of the pipeline
      // if the NEXT buffer we would write to is not ready for use, then
      // assert almost full
      almost_full_pipeline.last() =
          ready_to_send[(cur_rx_buffer + 1) & kNumScratchMemCopiesBitMask];

      // read the next data to send
      PipeType data_out;
      UnrolledLoop<k_pipe_width>([&](auto i) {
        data_out.template get<i>() = scratch[cur_tx_buffer][i][cur_tx_col];
      });

      // only attempt to send the data if it is valid
      if (ready_to_send[cur_tx_buffer]) {
        bool write_success;
        MatrixOutPipe::write(data_out, write_success);

        // update the transmit buffer status only if the pipe write succeeded
        if (write_success) {
          if (cur_last_tx_col) {
            ready_to_send[cur_tx_buffer] = false;
            tx_col = 0;
            tx_buffer = (cur_tx_buffer + 1) & kNumScratchMemCopiesBitMask;
            last_tx_col = false;
          } else {
            tx_col++;
            // if the current value of tx_col is 2 less than the total, then
            // the next value we read out (when cur_tx_col == k_num_cols_in -1)
            // is the last value for this copy
            last_tx_col = (cur_tx_col == (unsigned short)k_num_cols_in - 2);
          }
        }
      }

      // as long as the internal buffers are not almost full, read new data
      PipeType data_in;
      bool read_valid = false;
      if (!cur_almost_full) {
        data_in = MatrixInPipe::read(read_valid);
      }

      // if we have new data, store it in the buffer and update the status
      if (read_valid) {
        unsigned short row = (cur_rx_count * (unsigned short)k_pipe_width) /
                             (unsigned short)k_num_cols_in;
        unsigned short col = (cur_rx_count * (unsigned short)k_pipe_width) %
                             (unsigned short)k_num_cols_in;
        UnrolledLoop<k_pipe_width>([&](auto i) {
          scratch[cur_rx_buffer][row][col + i] = data_in.template get<i>();
        });

        // update the receive buffer status
        if (cur_rx_count == (unsigned short)((k_num_cols_in - 1))) {
          ready_to_send[cur_rx_buffer] = true;
          rx_count = 0;
          rx_buffer = (cur_rx_buffer + 1) & kNumScratchMemCopiesBitMask;
        } else {
          rx_count++;
        }
      }

    }  // end of while(1)
  }    // end of operator()()
};

// Special case for a k_pipe_width=1
// In this case, the is just a pass through kernel since the matrix is
// 1 x k_num_cols. Overriding this version allows us to save area.
template <typename T, size_t k_num_cols_in, typename MatrixInPipe,
          typename MatrixOutPipe>
struct Transposer<T, k_num_cols_in, 1, MatrixInPipe, MatrixOutPipe> {
  void operator()() const {
    while (1) {
      auto d = MatrixInPipe::read();
      MatrixOutPipe::write(d);
    }
  }
};

#endif  // ifndef __TRANSPOSE_HPP__
