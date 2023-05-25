#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

namespace fpga_linalg {

/**
 * Matrix multiply kernel.
 *
 * Repeatedly reads matrix tiles of A and B from input pipes and computes A * B
 * using a systolic array of PEs. Writes result matrix tile of C to output pipe.
 *
 */
template <typename TT,       // Datatype of the elements of the matrix
          int common,        // Columns of matrix A / rows of matrix B
          int tile_a,        // Tile size for matrix A
          int tile_b,        // Tile size for matrix B
          typename PipeA,    // Input pipe for matrix A
          typename PipeB,    // Input pipe for matrix B
          typename PipeC,    // Output pipe for matrix C
          typename PipeDone> // Pipe to receive signal to stop reading inputs
class StreamingMatmul {
public:
  void operator()() const {
    // An array of registers to accumulate the dot products which form the
    // output matrix C; one register per PE; initialized to 0 in order to infer
    // the FP accumulator
    [[intel::fpga_register]] // NO-FORMAT: Attribute
    TT accum[tile_a][tile_b];

    fpga_tools::UnrolledLoop<tile_a>([&](auto row) {
      fpga_tools::UnrolledLoop<tile_b>([&](auto col) {
        accum[row][col] = 0;
      });
    });

    // Matrix is flushed out to a secondary register storage when it is fully
    // computed, from where it is streamed out to the pipe
    [[intel::fpga_register]] // NO-FORMAT: Attribute
    TT results[tile_a][tile_b];

    constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<common + 1>();
    ac_int<kCommonBitSize, false> counter = 0;
    bool write_flag = false;
    bool last_pipe_read = false;

    // Compute matrix multiplications as long as matrices are given as inputs
    [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
    while (1) {
      // Between matrices and/or matrix tiles, reset the accumulators to 0
      if (counter == 0) {
        fpga_tools::UnrolledLoop<tile_a>([&](auto row) {
          fpga_tools::UnrolledLoop<tile_b>([&](auto col) {
            accum[row][col] = 0;
          });
        });
      }

      // Read matrices A and B from the two input pipes; feeder A will send a
      // signal when there are no more matrices to compute, at which point we
      // should stop reading inputs
      fpga_tools::NTuple<TT, tile_a> pipe_read_a;
      fpga_tools::NTuple<TT, tile_b> pipe_read_b;
      fpga_tools::UnrolledLoop<tile_a>([&](auto row) {
        pipe_read_a.template get<row>() = 0;
      });
      fpga_tools::UnrolledLoop<tile_b>([&](auto col) {
        pipe_read_b.template get<col>() = 0;
      });
      if (!last_pipe_read) {
        pipe_read_a = PipeA::read();
        pipe_read_b = PipeB::read();
        last_pipe_read = PipeDone::read();
      }

      // Compute the matrix product; fully unrolled loop to describe an array of
      // processing elements
      fpga_tools::UnrolledLoop<tile_a>([&](auto row) {
        fpga_tools::UnrolledLoop<tile_b>([&](auto col) {
          pipe_read_a.template get<row>() =
              sycl::ext::intel::fpga_reg(pipe_read_a.template get<row>());
          pipe_read_b.template get<col>() =
              sycl::ext::intel::fpga_reg(pipe_read_b.template get<col>());
          TT result = pipe_read_a.template get<row>() *
                      pipe_read_b.template get<col>() + accum[row][col];
          accum[row][col] = result;
          // Flush matrix to results array if finished computing
          if (counter == (common - 1)) {
            results[row][col] = result;
            write_flag = true;
          }
        });
      });

      if (counter == common - 1) {
        counter = 0;
      } else {
        counter++;
      }

      // Stop writing while we wait for the next matrix to finish computing
      // (only necessary if common > tile_b, i.e., when it takes strictly longer
      // to compute than to write to pipe)
      if constexpr (common > tile_b) {
        if ((counter == tile_b) & write_flag) {
          write_flag = false;
        }
      }

      // Write the result matrix C from the registers to the output pipe
      if (write_flag) {
        fpga_tools::NTuple<TT, tile_a> pipe_write;
        fpga_tools::UnrolledLoop<tile_a>([&](auto row) {
          pipe_write.template get<row>() = results[row][0];
          fpga_tools::UnrolledLoop<tile_b - 1>([&](auto k) {
            results[row][k] = results[row][k + 1];
          });
        });
        PipeC::write(pipe_write);
      }
    } // end of while (1)
  }   // end of operator
};

} // namespace fpga_linalg

#endif /* __STREAMING_MATMUL_HPP__ */