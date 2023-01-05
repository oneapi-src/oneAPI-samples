#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;
using namespace fpga_tools;

template <typename T, int common, int tile_common, int tile_A, int tile_B,
          int pipe_size, typename pipe_A, typename pipe_B, typename pipe_C>
void streaming_matmul() {
  
  // Iterations to process a row / column
  constexpr bool incomplete_burst_A = tile_A % pipe_size != 0;
  constexpr bool incomplete_burst_B = tile_B % pipe_size != 0;
  constexpr int rw_iter_A = tile_A / pipe_size + (incomplete_burst_A ? 1 : 0);
  constexpr int rw_iter_B = tile_B / pipe_size + (incomplete_burst_B ? 1 : 0);
  constexpr int iters = std::max(rw_iter_A, rw_iter_B);

  // Iterations to process a tile
  constexpr int num_tiles = common / tile_common;
  constexpr int common_bitsize = BitsForMaxValue<tile_common + 1>();

  // Iterations to load matrix from pipe
  constexpr int feeder_iters = tile_common * iters;
  constexpr int feeder_iters_bitsize = BitsForMaxValue<feeder_iters + 1>();

  // Iterations to store matrix to pipe
  constexpr int drain_iters = tile_B * rw_iter_A;
  constexpr int drain_iters_bitsize = BitsForMaxValue<drain_iters + 1>();

  // Memory attributes
  constexpr short bank_width = pipe_size * sizeof(T);
  constexpr int num_banks_A = tile_A / pipe_size;
  constexpr int num_banks_B = tile_B / pipe_size;
  constexpr int num_banks_A_pow2 = Pow2(CeilLog2(num_banks_A));
  constexpr int num_banks_B_pow2 = Pow2(CeilLog2(num_banks_B));

  while (1) {

    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    T accum[tile_A][tile_B];

    UnrolledLoop<tile_A>([&](auto row) {
      UnrolledLoop<tile_B>([&](auto col) {
        accum[row][col] = 0;
      });
    });

    for (int tile = 0; tile < num_tiles; tile++) {
      
      [[intel::numbanks(num_banks_A_pow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(bank_width)]]       // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]           // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]           // NO-FORMAT: Attribute
      T mem_A[tile_common][tile_A];

      [[intel::numbanks(num_banks_B_pow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(bank_width)]]       // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]           // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]           // NO-FORMAT: Attribute
      T mem_B[tile_common][tile_B];

      // Copy a matrix from the pipe to a local memory
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<feeder_iters_bitsize, false> i = 0; i < feeder_iters; i++) {
        int row_iter = (int)(i) % iters;
        int col_iter = (int)(i) / iters;

        NTuple<T, pipe_size> pipe_read_A;
        NTuple<T, pipe_size> pipe_read_B;
        if (row_iter < rw_iter_A) {
          pipe_read_A = pipe_A::read();
        }
        if (row_iter < rw_iter_B) {
          pipe_read_B = pipe_B::read();
        }

        UnrolledLoop<iters>([&](auto k) {
          UnrolledLoop<pipe_size>([&](auto t) {
            constexpr int idx = k * pipe_size + t;
            if constexpr (idx < tile_A) {
              if (row_iter == k) {
                mem_A[col_iter][idx] = pipe_read_A.template get<t>();
              }
            }
            if constexpr (idx < tile_B) {
              if (row_iter == k) {
                mem_B[col_iter][idx] = pipe_read_B.template get<t>();
              }
            }
            pipe_read_A.template get<t>() =
                ext::intel::fpga_reg(pipe_read_A.template get<t>());
            pipe_read_B.template get<t>() =
                ext::intel::fpga_reg(pipe_read_B.template get<t>());
          });
          row_iter = ext::intel::fpga_reg(row_iter);
        });
      }

      // Compute the matrix product
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<common_bitsize, false> i = 0; i < tile_common; i++) {
        T fed_A[tile_A];
        T fed_B[tile_B];

        UnrolledLoop<tile_A>([&](auto row) { fed_A[row] = mem_A[i][row]; });
        UnrolledLoop<tile_B>([&](auto col) { fed_B[col] = mem_B[i][col]; });

        // Unrolled loop to describe an array of PEs
        UnrolledLoop<tile_A>([&](auto row) {
          UnrolledLoop<tile_B>([&](auto col) {
            fed_A[row] = ext::intel::fpga_reg(fed_A[row]);
            fed_B[col] = ext::intel::fpga_reg(fed_B[col]);
            accum[row][col] += fed_A[row] * fed_B[col];
          });
        });
      }
    }

    // Copy the result matrix on the output pipe
    [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
    [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
    for (ac_int<drain_iters_bitsize, false> i = 0; i < drain_iters; i++) {
      int row_iter = (int)(i) % rw_iter_A;
      int col_iter = (int)(i) / rw_iter_A;
      bool get[rw_iter_A];

      UnrolledLoop<rw_iter_A>([&](auto k) {
        get[k] = row_iter == k;
        row_iter = ext::intel::fpga_reg(row_iter);
      });

      NTuple<T, pipe_size> pipe_write;
      UnrolledLoop<rw_iter_A>([&](auto k) {
        UnrolledLoop<pipe_size>([&](auto t) {
          constexpr int idx = k * pipe_size + t;
          if constexpr (idx < tile_A) {
            pipe_write.template get<t>() =
                get[k] ? accum[idx][col_iter]
                       : ext::intel::fpga_reg(pipe_write.template get<t>());
          }
        });
      });

      pipe_C::write(pipe_write);
    }
  }
}

#endif /* __STREAMING_MATMUL_HPP__ */