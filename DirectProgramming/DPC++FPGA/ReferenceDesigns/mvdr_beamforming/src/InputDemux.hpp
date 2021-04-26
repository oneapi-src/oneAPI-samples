#ifndef __INPUT_DEMUX_HPP__
#define __INPUT_DEMUX_HPP__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <cmath>

// utility classes
#include "Tuple.hpp"
#include "UnrolledLoop.hpp"

#include "mvdr_complex.hpp"

using namespace sycl;

// SubmitInputDemuxKernel
// Accept data from an input pipe, and split the data into two output pipes
// based on header packets detected in the data stream.  Buffer up a full
// training array and matching data samples before passing any data downstream,
// so that the correct amount of each type of data is guaranteed to all
// downstream kernels, and no partial matrices are ever transferred.
template <typename InputDemuxKernelName,  // Name to use for the Kernel

          size_t k_pipe_width,         // Number of complex numbers transferred
                                       // on each pipe read or write
          size_t k_training_size,      // number of complex numbers in a full
                                       // training matrix
          size_t k_max_xrx_data_size,  // maximum number of complex numbers in a
                                       // set of xrx data to be matched with
                                       // each training matrix to support
          bool k_read_every_cycle,  // When true, kernel will try to read from
                                    // input pipe on every cycle.  When false,
                                    // only read when space is available.  Set
                                    // to false for backpressurable interfaces

          typename DataInPipe,           // Receive streaming data, including
                                         // headers, from this pipe.
          typename TrainingDataOutPipe,  // Send training data to QRD
          typename XrxDataOutPipe        // Send sample data to Beamformer
          >
event SubmitInputDemuxKernel(
    queue& q,
    int xrx_data_size  // number of complex numbers in a set of
                       // xrx data to be matched with each
                       // training matrix
                       // (must be <= k_max_xrx_data_size)
) {
  // error checking on input parameter
  if (xrx_data_size > k_max_xrx_data_size) {
    std::cerr << "Called SubmitInputDemuxKernel() with k_max_xrx_data_size < ";
    std::cerr << "xrx_data_size" << std::endl;
    std::terminate();
  }

  // Use an NTuple of complex numbers for reading/writing pipes
  using PipeType = NTuple<ComplexType, k_pipe_width>;

  auto e = q.submit([&](handler& h) {
    h.single_task<InputDemuxKernelName>([=] {
      constexpr int kReadsPerTrainingMatrix = k_training_size / k_pipe_width;
      constexpr int kMaxReadsPerXrxDataMatrix =
          k_max_xrx_data_size / k_pipe_width;
      int reads_per_xrx_matrix = xrx_data_size / k_pipe_width;

      // multiple copies of each matrix to create a simple FIFO that has enough
      // slack to allow us to use 'almost full' in the logic below
      constexpr unsigned char kNumMatrixCopies = 4;  // must be power of 2
      static_assert(kNumMatrixCopies > 0);
      static_assert((kNumMatrixCopies & (kNumMatrixCopies - 1)) == 0);
      constexpr unsigned char kNumMatrixCopiesBitMask = kNumMatrixCopies - 1;
      PipeType training_matrix[kNumMatrixCopies][kReadsPerTrainingMatrix];
      PipeType xrx_data_matrix[kNumMatrixCopies][kMaxReadsPerXrxDataMatrix];

      // track the status of each of the buffers
      // NO-FORMAT comments are for clang-format
      [[intel::fpga_register]]  // NO-FORMAT: Attribute
      bool ready_to_send[kNumMatrixCopies] = {false};

      // create a 'pipeline' for the almost full signal
      constexpr int kAlmostFullPipeDepth = 2;
      NTuple<bool, kAlmostFullPipeDepth> almost_full_pipeline;
      UnrolledLoop<kAlmostFullPipeDepth>([&](auto pipe_stage) {
        almost_full_pipeline.template get<pipe_stage>() = false;
      });

      // state variables for receiving data
      unsigned char rx_buffer = 0;  // which buffer are we currently writing
      unsigned int rx_training_count = 0;
      unsigned int rx_xrx_count = 0;
      enum class RxState : char {
        wait_training_header,     // discard data until training header
        receiving_training_data,  // store training data
        expect_xrx_data_header,   // next word should be xrx data header
        receiving_xrx_data        // store xrx data
      };
      RxState rx_state = RxState::wait_training_header;

      // state variables for sending data
      unsigned char tx_buffer = 0;  // which buffer are we currently reading
      unsigned int tx_training_count = 0;
      unsigned int tx_xrx_count = 0;
      bool all_training_sent = false;
      bool all_xrx_sent = false;

      // NO-FORMAT comments are for clang-format
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      while (1) {
        // capture the current state of variables before they are modified
        // by this iteration of the loop
        unsigned char cur_rx_buffer = rx_buffer;
        unsigned int cur_rx_training_count = rx_training_count;
        unsigned int cur_rx_xrx_count = rx_xrx_count;
        RxState cur_rx_state = rx_state;
        unsigned char cur_tx_buffer = tx_buffer;
        unsigned int cur_tx_training_count = tx_training_count;
        unsigned int cur_tx_xrx_count = tx_xrx_count;
        bool cur_all_training_sent = all_training_sent;
        bool cur_all_xrx_sent = all_xrx_sent;

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
            ready_to_send[(cur_rx_buffer + 1) & kNumMatrixCopiesBitMask];

        // Only do the pipe read if we have space available, OR if we are
        // configured to read on every cycle.  Reading on every cycle means
        // this block will discard data if downstream can't keep up (and
        // therefore be able to track how much data is discarded).
        PipeType data_in;
        bool read_valid = false;
        if (k_read_every_cycle || !cur_almost_full) {
          data_in = DataInPipe::read(read_valid);
        }

        // examine received data for header markers (ignored if !read_valid)
        // Real part of 0th word set to NaN indicates a header
        // Imaginary part of 0th word set to 0 indicates training header, non
        // zero value indicates xrx data header
        bool is_header = std::isnan(data_in.template get<0>().real());
        bool header_is_training = (data_in.template get<0>().imag() == 0.0);

        // only process valid data if we are not almost full, otherwise data
        // must be discarded
        if (read_valid && !cur_almost_full) {
          // store training data
          if (cur_rx_state == RxState::receiving_training_data) {
            training_matrix[cur_rx_buffer][cur_rx_training_count] = data_in;
            rx_training_count++;
          } else {
            rx_training_count = 0;
          }

          // store xrx data
          if (cur_rx_state == RxState::receiving_xrx_data) {
            xrx_data_matrix[cur_rx_buffer][cur_rx_xrx_count] = data_in;
            rx_xrx_count++;
          } else {
            rx_xrx_count = 0;
          }

          // update the rx_buffer when all data has been received
          if (cur_rx_state == RxState::receiving_xrx_data &&
              cur_rx_xrx_count == reads_per_xrx_matrix - 1) {
            ready_to_send[cur_rx_buffer] = true;
            rx_buffer = (rx_buffer + 1) & kNumMatrixCopiesBitMask;
          }

          // Rx state machine
          if (cur_rx_state == RxState::wait_training_header) {
            if (is_header && header_is_training) {
              rx_state = RxState::receiving_training_data;
            }
          } else if (cur_rx_state == RxState::receiving_training_data) {
            if (is_header) {
              // unexpected header, discard the current data and start over
              rx_state = RxState::wait_training_header;
            } else if (cur_rx_training_count == kReadsPerTrainingMatrix - 1) {
              rx_state = RxState::expect_xrx_data_header;
            }
          } else if (cur_rx_state == RxState::expect_xrx_data_header) {
            if (is_header && !header_is_training) {
              rx_state = RxState::receiving_xrx_data;
            } else {
              // didn't receive expected header, discard data and start over
              rx_state = RxState::wait_training_header;
            }
          } else {  // RxState::receiving_xrx_data
            if (is_header) {
              // unexpected header, discard the current data and start over
              rx_state = RxState::wait_training_header;
            } else if (cur_rx_xrx_count == reads_per_xrx_matrix - 1) {
              rx_state = RxState::wait_training_header;
            }
          }  // end of Rx state machine

        } else if (read_valid) {
          // if we have valid data but are almost full, we must discard all data
          rx_training_count = 0;
          rx_xrx_count = 0;
          rx_state = RxState::wait_training_header;
        }

        // try to send data unless we have sent all data from the current buffer
        if (!cur_all_training_sent || !cur_all_xrx_sent) {
          bool write_success = false;

          // send training data, if available
          if (ready_to_send[cur_tx_buffer] && !cur_all_training_sent) {
            TrainingDataOutPipe::write(
                training_matrix[cur_tx_buffer][cur_tx_training_count],
                write_success);
          }
          
          if (write_success) {
            all_training_sent =
                (cur_tx_training_count == kReadsPerTrainingMatrix - 1);
            tx_training_count++;
          }

          // send xrx data, if available
          if (ready_to_send[cur_tx_buffer] && !cur_all_xrx_sent) {
            XrxDataOutPipe::write(
                xrx_data_matrix[cur_tx_buffer][cur_tx_xrx_count],
                write_success);
          } else {
            write_success = false;
          }
          if (write_success) {
            all_xrx_sent = (cur_tx_xrx_count == reads_per_xrx_matrix - 1);
            tx_xrx_count++;
          }

        } else {
          // all training and xrx data from the current buffer has been sent
          ready_to_send[cur_tx_buffer] = false;
          tx_buffer = (tx_buffer + 1) & kNumMatrixCopiesBitMask;
          all_training_sent = false;
          all_xrx_sent = false;
          tx_training_count = 0;
          tx_xrx_count = 0;
        }

      }  // end of while( 1 )
    });  // end of h.single_task
  });    // end of q.submit

  return e;

}  // end of SubmitInputDemuxKernel()

#endif  // ifndef __INPUT_DEMUX_HPP__
