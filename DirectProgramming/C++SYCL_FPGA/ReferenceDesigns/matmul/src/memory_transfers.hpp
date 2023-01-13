#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using BurstCoalescedLSU =
    sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>>;

template <typename T, int k_rows_a, int k_common, int k_cols_b, int k_tile_a,
          int k_tile_b, int k_pipe_size, typename PipeA>
void MatrixReadFromDDRToPipeA(T *a, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstA = k_tile_a % k_pipe_size != 0;

  // Number of pipe reads to read a full column of A and number of columns
  constexpr int kRWIterA = k_tile_a / k_pipe_size + (kIncompleteBurstA ? 1 : 0);
  constexpr int kRWIterABitSize = fpga_tools::BitsForMaxValue<kRWIterA + 1>();
  constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<k_common + 1>();

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Size of a full matrix A
  constexpr int kMatsizeA = k_rows_a * k_common;

  sycl::device_ptr<T> a_ptr(a);

  // Repeatedly read matrix tiles from global memory and send them to the pipe
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::loop_coalesce(4)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kBlocksABitSize, false> block_A = 0; block_A < kBlocksA;
           block_A++) {
        for (ac_int<kBlocksBBitSize, false> block_B = 0; block_B < kBlocksB;
             block_B++) {
          for (ac_int<kCommonBitSize, false> i = 0; i < k_common; i++) {
            for (ac_int<kRWIterABitSize, false> j = 0; j < kRWIterA; j++) {

              // Perform the burst read of k_pipe_size elements
              fpga_tools::NTuple<T, k_pipe_size> pipe_write;
              fpga_tools::UnrolledLoop<k_pipe_size>([&](auto k) {
                int idx = (mat * kMatsizeA) + ((int)(block_A)*k_tile_a) +
                          ((int)(i)*k_rows_a) + ((int)(j)*k_pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (kIncompleteBurstA) {
                  if ((j * k_pipe_size + k) < k_tile_a) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(a_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(a_ptr + idx);
                }
              });

              PipeA::write(pipe_write);
            }
          }
        }
      }
    }
  }
}

template <typename T, int k_rows_a, int k_common, int k_cols_b, int k_tile_a,
          int k_tile_b, int k_pipe_size, typename PipeB>
void MatrixReadFromDDRToPipeB(T *b, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstB = k_tile_b % k_pipe_size != 0;

  // Number of pipe reads to read a full row of B and number of rows
  constexpr int kRWIterB = k_tile_b / k_pipe_size + (kIncompleteBurstB ? 1 : 0);
  constexpr int kRWIterBBitSize = fpga_tools::BitsForMaxValue<kRWIterB + 1>();
  constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<k_common + 1>();

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Size of a full matrix B
  constexpr int kMatsizeB = k_cols_b * k_common;

  sycl::device_ptr<T> b_ptr(b);

  // Repeatedly read matrix tiles from global memory and send them to the pipe
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::loop_coalesce(4)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kBlocksABitSize, false> block_A = 0; block_A < kBlocksA;
           block_A++) {
        for (ac_int<kBlocksBBitSize, false> block_B = 0; block_B < kBlocksB;
             block_B++) {
          for (ac_int<kCommonBitSize, false> i = 0; i < k_common; i++) {
            for (ac_int<kRWIterBBitSize, false> j = 0; j < kRWIterB; j++) {

              // Perform the burst read of k_pipe_size elements
              fpga_tools::NTuple<T, k_pipe_size> pipe_write;
              fpga_tools::UnrolledLoop<k_pipe_size>([&](auto k) {
                int idx = (mat * kMatsizeB) + ((int)(block_B)*k_tile_b) +
                          ((int)(i)*k_cols_b) + ((int)(j)*k_pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (kIncompleteBurstB) {
                  if ((j * k_pipe_size + k) < k_tile_b) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(b_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(b_ptr + idx);
                }
              });

              PipeB::write(pipe_write);
            }
          }
        }
      }
    }
  }
}

template <typename T, int k_rows_a, int k_cols_b, int k_tile_a, int k_tile_b,
          int k_pipe_size, typename PipeC>
void MatrixReadPipeToDDR(T *c, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstA = k_tile_a % k_pipe_size != 0;

  // Number of pipe reads to read a full column of C
  constexpr int kRWIterA = k_tile_a / k_pipe_size + (kIncompleteBurstA ? 1 : 0);

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Iterations to store matrix to global memory
  constexpr int kDrainIters = k_tile_b * kRWIterA;
  constexpr int kDrainItersBitSize =
      fpga_tools::BitsForMaxValue<kDrainIters + 1>();

  // Size of a full matrix C
  constexpr int kMatsizeC = k_rows_a * k_cols_b;

  sycl::device_ptr<T> c_ptr(c);

  // Repeatedly read matrix tiles from pipe and write them to global memory
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::loop_coalesce(3)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kBlocksABitSize, false> block_A = 0; block_A < kBlocksA;
           block_A++) {
        for (ac_int<kBlocksBBitSize, false> block_B = 0; block_B < kBlocksB;
             block_B++) {
          for (ac_int<kDrainItersBitSize, false> i = 0; i < kDrainIters; i++) {

            int row_iter = (int)(i) % kRWIterA;
            int col_iter = (int)(i) / kRWIterA;

            // Perform the burst write of k_pipe_size elements
            fpga_tools::NTuple<T, k_pipe_size> pipe_read = PipeC::read();
            fpga_tools::UnrolledLoop<k_pipe_size>([&](auto k) {
              int idx = (mat * kMatsizeC) +
                        ((int)(block_B)*k_tile_b * k_rows_a) +
                        ((int)(block_A)*k_tile_a) + (col_iter * k_rows_a) +
                        (row_iter * k_pipe_size + k);
              // Only perform the writes that are relevant (and don't access a
              // memory address that may be beyond the matrix last address)
              if constexpr (kIncompleteBurstA) {
                if ((row_iter * k_pipe_size + k) < k_tile_a) {
                  BurstCoalescedLSU::store(c_ptr + idx,
                                           pipe_read.template get<k>());
                }
              } else {
                BurstCoalescedLSU::store(c_ptr + idx,
                                         pipe_read.template get<k>());
              }
            });
          }
        }
      }
    }
  }
}

#endif /* __MEMORY_TRANSFERS_HPP__ */