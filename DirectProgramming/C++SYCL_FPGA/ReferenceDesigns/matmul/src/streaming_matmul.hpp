#ifndef __STREAMING_MATMUL_HPP__
#define __STREAMING_MATMUL_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <typename T, int k_common, int k_tile_common, int k_tile_a,
          int k_tile_b, int k_pipe_size, typename PipeA, typename PipeB,
          typename PipeC>
void StreamingMatmul() {
  
  // Iterations to process a row / column
  constexpr bool kIncompleteBurstA = k_tile_a % k_pipe_size != 0;
  constexpr bool kIncompleteBurstB = k_tile_b % k_pipe_size != 0;
  constexpr int kRWIterA = k_tile_a / k_pipe_size + (kIncompleteBurstA ? 1 : 0);
  constexpr int kRWIterB = k_tile_b / k_pipe_size + (kIncompleteBurstB ? 1 : 0);
  constexpr int kIters = std::max(kRWIterA, kRWIterB);

  // Iterations to process a tile
  constexpr int kNumTiles = k_common / k_tile_common;
  constexpr int kCommonBitSize =
      fpga_tools::BitsForMaxValue<k_tile_common + 1>();

  // Iterations to load matrix from pipe
  constexpr int kFeederIters = k_tile_common * kIters;
  constexpr int kFeederItersBitSize =
      fpga_tools::BitsForMaxValue<kFeederIters + 1>();

  // Iterations to store matrix to pipe
  constexpr int kDrainIters = k_tile_b * kRWIterA;
  constexpr int kDrainItersBitSize =
      fpga_tools::BitsForMaxValue<kDrainIters + 1>();

  // Memory attributes
  constexpr short kBankWidth = k_pipe_size * sizeof(T);
  constexpr int kNumBanksA = k_tile_a / k_pipe_size;
  constexpr int kNumBanksB = k_tile_b / k_pipe_size;
  constexpr int kNumBanksAPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanksA));
  constexpr int kNumBanksBPow2 =
      fpga_tools::Pow2(fpga_tools::CeilLog2(kNumBanksB));

  while (1) {

    [[intel::fpga_register]]  // NO-FORMAT: Attribute
    T accum[k_tile_a][k_tile_b];

    fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
      fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
        accum[row][col] = 0;
      });
    });

    for (int tile = 0; tile < kNumTiles; tile++) {
      
      [[intel::numbanks(kNumBanksAPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankWidth)]]     // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]         // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]         // NO-FORMAT: Attribute
      T mem_a[k_tile_common][k_tile_a];

      [[intel::numbanks(kNumBanksBPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankWidth)]]     // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]         // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]         // NO-FORMAT: Attribute
      T mem_b[k_tile_common][k_tile_b];

      // Copy a matrix from the pipe to a local memory
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kFeederItersBitSize, false> i = 0; i < kFeederIters; i++) {
        int row_iter = (int)(i) % kIters;
        int col_iter = (int)(i) / kIters;

        fpga_tools::NTuple<T, k_pipe_size> pipe_read_a;
        fpga_tools::NTuple<T, k_pipe_size> pipe_read_b;
        if (row_iter < kRWIterA) {
          pipe_read_a = PipeA::read();
        }
        if (row_iter < kRWIterB) {
          pipe_read_b = PipeB::read();
        }

        fpga_tools::UnrolledLoop<kIters>([&](auto k) {
          fpga_tools::UnrolledLoop<k_pipe_size>([&](auto t) {
            constexpr int kIdx = k * k_pipe_size + t;
            if constexpr (kIdx < k_tile_a) {
              if (row_iter == k) {
                mem_a[col_iter][kIdx] = pipe_read_a.template get<t>();
              }
            }
            if constexpr (kIdx < k_tile_b) {
              if (row_iter == k) {
                mem_b[col_iter][kIdx] = pipe_read_b.template get<t>();
              }
            }
            pipe_read_a.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_a.template get<t>());
            pipe_read_b.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_b.template get<t>());
          });
          row_iter = sycl::ext::intel::fpga_reg(row_iter);
        });
      }

      // Compute the matrix product
      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kCommonBitSize, false> i = 0; i < k_tile_common; i++) {
        T fed_A[k_tile_a];
        T fed_B[k_tile_b];

        fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
          fed_A[row] = mem_a[i][row];
        });
        fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
          fed_B[col] = mem_b[i][col];
        });

        // Unrolled loop to describe an array of PEs
        fpga_tools::UnrolledLoop<k_tile_a>([&](auto row) {
          fpga_tools::UnrolledLoop<k_tile_b>([&](auto col) {
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

      fpga_tools::NTuple<T, k_pipe_size> pipe_write;
      fpga_tools::UnrolledLoop<kRWIterA>([&](auto k) {
        fpga_tools::UnrolledLoop<k_pipe_size>([&](auto t) {
          constexpr int kIdx = k * k_pipe_size + t;
          if constexpr (kIdx < k_tile_a) {
            pipe_write.template get<t>() =
                get[k]
                    ? accum[kIdx][col_iter]
                    : sycl::ext::intel::fpga_reg(pipe_write.template get<t>());
          }
        });
      });

      PipeC::write(pipe_write);
    }
  }
}

#endif /* __STREAMING_MATMUL_HPP__ */