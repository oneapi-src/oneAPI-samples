#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <typename T, int common, int tile_common, int tile_A, int tile_B,
          int pipe_size, typename pipe_A, typename pipe_B, typename pipe_C>
void streaming_matmul() {
  
  // Iterations to process a row / column
  constexpr bool kIncompleteBurstA = tile_A % pipe_size != 0;
  constexpr bool kIncompleteBurstB = tile_B % pipe_size != 0;
  constexpr int kRWIterA = tile_A / pipe_size + (kIncompleteBurstA ? 1 : 0);
  constexpr int kRWIterB = tile_B / pipe_size + (kIncompleteBurstB ? 1 : 0);
  constexpr int kIters = std::max(kRWIterA, kRWIterB);

  // Iterations to process a tile
  constexpr int kNumTiles = common / tile_common;
  constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<tile_common + 1>();

  // Iterations to load matrix from pipe
  constexpr int kFeederIters = tile_common * kIters;
  constexpr int kFeederItersBitSize =
      fpga_tools::BitsForMaxValue<kFeederIters + 1>();

  // Iterations to store matrix to pipe
  constexpr int kDrainIters = tile_B * kRWIterA;
  constexpr int kDrainItersBitSize =
      fpga_tools::BitsForMaxValue<kDrainIters + 1>();

  // Memory attributes
  constexpr short kBankWidth = pipe_size * sizeof(T);
  constexpr int kNumBanksA = tile_A / pipe_size;
  constexpr int kNumBanksB = tile_B / pipe_size;
  constexpr int kNumBanksAPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanksA));
  constexpr int kNumBanksBPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanksB));

  while (1) {
    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    T accum[tile_A][tile_B];

    fpga_tools::UnrolledLoop<tile_A>([&](auto row) {
      fpga_tools::UnrolledLoop<tile_B>([&](auto col) {
        accum[row][col] = 0;
      });
    });

    for (int tile = 0; tile < kNumTiles; tile++) {
      [[intel::numbanks(kNumBanksAPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankWidth)]]     // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]         // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]         // NO-FORMAT: Attribute
      T mem_A[tile_common][tile_A];

      [[intel::numbanks(kNumBanksBPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankWidth)]]     // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]         // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]         // NO-FORMAT: Attribute
      T mem_B[tile_common][tile_B];

      // Copy a matrix from the pipe to a local memory
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kFeederItersBitSize, false> i = 0; i < kFeederIters; i++) {
        int row_iter = (int)(i) % kIters;
        int col_iter = (int)(i) / kIters;

        fpga_tools::NTuple<T, pipe_size> pipe_read_A;
        fpga_tools::NTuple<T, pipe_size> pipe_read_B;
        if (row_iter < kRWIterA) {
          pipe_read_A = pipe_A::read();
        }
        if (row_iter < kRWIterB) {
          pipe_read_B = pipe_B::read();
        }

        fpga_tools::UnrolledLoop<kIters>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            constexpr int kIdx = k * pipe_size + t;
            if constexpr (kIdx < tile_A) {
              if (row_iter == k) {
                mem_A[col_iter][kIdx] = pipe_read_A.template get<t>();
              }
            }
            if constexpr (kIdx < tile_B) {
              if (row_iter == k) {
                mem_B[col_iter][kIdx] = pipe_read_B.template get<t>();
              }
            }
            pipe_read_A.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_A.template get<t>());
            pipe_read_B.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_B.template get<t>());
          });
          row_iter = sycl::ext::intel::fpga_reg(row_iter);
        });
      }

      // Compute the matrix product
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kCommonBitSize, false> i = 0; i < tile_common; i++) {
        T fed_A[tile_A];
        T fed_B[tile_B];

        fpga_tools::UnrolledLoop<tile_A>([&](auto row) {
          fed_A[row] = mem_A[i][row];
        });
        fpga_tools::UnrolledLoop<tile_B>([&](auto col) {
          fed_B[col] = mem_B[i][col];
        });

        // Unrolled loop to describe an array of PEs
        fpga_tools::UnrolledLoop<tile_A>([&](auto row) {
          fpga_tools::UnrolledLoop<tile_B>([&](auto col) {
            fed_A[row] = sycl::ext::intel::fpga_reg(fed_A[row]);
            fed_B[col] = sycl::ext::intel::fpga_reg(fed_B[col]);
            accum[row][col] += fed_A[row] * fed_B[col];
          });
        });
      }
    }

    // Copy the result matrix on the output pipe
    [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
    [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
    for (ac_int<kDrainItersBitSize, false> i = 0; i < kDrainIters; i++) {
      int row_iter = (int)(i) % kRWIterA;
      int col_iter = (int)(i) / kRWIterA;
      bool get[kRWIterA];

      fpga_tools::UnrolledLoop<kRWIterA>([&](auto k) {
        get[k] = row_iter == k;
        row_iter = sycl::ext::intel::fpga_reg(row_iter);
      });

      fpga_tools::NTuple<T, pipe_size> pipe_write;
      fpga_tools::UnrolledLoop<kRWIterA>([&](auto k) {
        fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
          constexpr int kIdx = k * pipe_size + t;
          if constexpr (kIdx < tile_A) {
            pipe_write.template get<t>() =
                get[k]
                    ? accum[kIdx][col_iter]
                    : sycl::ext::intel::fpga_reg(pipe_write.template get<t>());
          }
        });
      });

      pipe_C::write(pipe_write);
    }
  }
}

#endif /* __STREAMING_MATMUL_HPP__ */