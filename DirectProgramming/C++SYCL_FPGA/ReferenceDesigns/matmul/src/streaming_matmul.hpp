#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

/**
 * Matrix multiply kernel.
 * 
 * Repeatedly reads matrix tiles of A and B from input pipes and computes A * B
 * using a systolic array of PEs. Writes result matrix tile of C to output pipe.
 *
 */
template <typename T,     // Datatype of the elements of the matrix
          int k_common,   // Columns of matrix A / rows of matrix B
          int k_tile_a,   // Tile size for matrix A
          int k_tile_b,   // Tile size for matrix B
          typename PipeA, // Input pipe for matrix A
          typename PipeB, // Input pipe for matrix B
          typename PipeC> // Output pipe for matrix C
class StreamingMatmul {
public:
  void operator()() const {
    constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<k_common + 1>();
    constexpr int kTileBBitSize = fpga_tools::BitsForMaxValue<k_tile_b + 1>();

    // Compute matrix multiplications as long as matrices are given as inputs
    while (1) {

      // An array of registers to accumulate the dot products which form the
      // output matrix C; one register per PE; initialized to 0
      [[intel::fpga_register]] // NO-FORMAT: Attribute
      T accum[k_tile_a][k_tile_b];

      fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
        fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
          accum[row][col] = 0;
        });
      });

      // Read matrices A and B from the two input pipes and compute the matrix
      // product; store the result in registers
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kCommonBitSize, false> i = 0; i < k_common; i++) {
        fpga_tools::NTuple<T, k_tile_a> pipe_read_a = PipeA::read();
        fpga_tools::NTuple<T, k_tile_b> pipe_read_b = PipeB::read();

        T fed_A[k_tile_a];
        T fed_B[k_tile_b];

        fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
          fed_A[row] = pipe_read_a.template get<row>();
        });
        fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
          fed_B[col] = pipe_read_b.template get<col>();
        });

        // Fully unrolled loop to describe an array of PEs
        fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
          fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
            fed_A[row] = sycl::ext::intel::fpga_reg(fed_A[row]);
            fed_B[col] = sycl::ext::intel::fpga_reg(fed_B[col]);
            accum[row][col] += fed_A[row] * fed_B[col];
          });
        });
      }  // end of i

      // Write the result matrix C from the registers to the output pipe
      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kTileBBitSize, false> i = 0; i < k_tile_b; i++) {
        fpga_tools::NTuple<T, k_tile_a> pipe_write;
        fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
          pipe_write.template get<row>() = accum[row][0];
          fpga_tools::UnrolledLoop<k_tile_b - 1>([&](auto k) {
            accum[row][k] = accum[row][k + 1];
          });
        });
        PipeC::write(pipe_write);
      }  // end of i
    }    // end of while(1)
  }      // end of operator
};

#endif /* __STREAMING_MATMUL_HPP__ */
