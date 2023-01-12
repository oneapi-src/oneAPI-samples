#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using BurstCoalescedLSU =
    sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>>;

template <typename T, int rows_A, int common, int cols_B, int tile_A,
          int tile_B, int pipe_size, typename PipeA>
void FeederA(T *A, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstA = tile_A % pipe_size != 0;

  // Number of pipe reads to read a full column of A and number of columns
  constexpr int kRWIterA = tile_A / pipe_size + (kIncompleteBurstA ? 1 : 0);
  constexpr int kRWIterABitSize = fpga_tools::BitsForMaxValue<kRWIterA + 1>();
  constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<common + 1>();

  // Number of tiles
  constexpr int kBlocksA = rows_A / tile_A;
  constexpr int kBlocksB = cols_B / tile_B;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Size of a full matrix A
  constexpr int kMatsizeA = rows_A * common;

  sycl::device_ptr<T> A_ptr(A);

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
          for (ac_int<kCommonBitSize, false> i = 0; i < common; i++) {
            for (ac_int<kRWIterABitSize, false> j = 0; j < kRWIterA; j++) {

              // Perform the burst read of pipe_size elements
              fpga_tools::NTuple<T, pipe_size> pipe_write;
              fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                int idx = (mat * kMatsizeA) + ((int)(block_A)*tile_A) +
                          ((int)(i)*rows_A) + ((int)(j)*pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (kIncompleteBurstA) {
                  if ((j * pipe_size + k) < tile_A) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(A_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(A_ptr + idx);
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

template <typename T, int rows_A, int common, int cols_B, int tile_A,
          int tile_B, int pipe_size, typename PipeB>
void FeederB(T *B, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstB = tile_B % pipe_size != 0;

  // Number of pipe reads to read a full row of B and number of rows
  constexpr int kRWIterB = tile_B / pipe_size + (kIncompleteBurstB ? 1 : 0);
  constexpr int kRWIterBBitSize = fpga_tools::BitsForMaxValue<kRWIterB + 1>();
  constexpr int kCommonBitSize = fpga_tools::BitsForMaxValue<common + 1>();

  // Number of tiles
  constexpr int kBlocksA = rows_A / tile_A;
  constexpr int kBlocksB = cols_B / tile_B;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Size of a full matrix B
  constexpr int kMatsizeB = cols_B * common;

  sycl::device_ptr<T> B_ptr(B);

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
          for (ac_int<kCommonBitSize, false> i = 0; i < common; i++) {
            for (ac_int<kRWIterBBitSize, false> j = 0; j < kRWIterB; j++) {

              // Perform the burst read of pipe_size elements
              fpga_tools::NTuple<T, pipe_size> pipe_write;
              fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
                int idx = (mat * kMatsizeB) + ((int)(block_B)*tile_B) +
                          ((int)(i)*cols_B) + ((int)(j)*pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (kIncompleteBurstB) {
                  if ((j * pipe_size + k) < tile_B) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(B_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(B_ptr + idx);
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

template <typename T, int rows_A, int cols_B, int tile_A, int tile_B,
          int pipe_size, typename PipeC>
void Drain(T *C, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool kIncompleteBurstA = tile_A % pipe_size != 0;

  // Number of pipe reads to read a full column of C
  constexpr int kRWIterA = tile_A / pipe_size + (kIncompleteBurstA ? 1 : 0);

  // Number of tiles
  constexpr int kBlocksA = rows_A / tile_A;
  constexpr int kBlocksB = cols_B / tile_B;
  constexpr int kBlocksABitSize = fpga_tools::BitsForMaxValue<kBlocksA + 1>();
  constexpr int kBlocksBBitSize = fpga_tools::BitsForMaxValue<kBlocksB + 1>();

  // Iterations to store matrix to global memory
  constexpr int kDrainIters = tile_B * kRWIterA;
  constexpr int kDrainItersBitSize =
      fpga_tools::BitsForMaxValue<kDrainIters + 1>();

  // Size of a full matrix C
  constexpr int kMatsizeC = rows_A * cols_B;

  sycl::device_ptr<T> C_ptr(C);

  // Repeatedly read matrix tiles from pipe and write them to global memory
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::ivdep]]                     // NO-FORMAT: Attribute
      [[intel::loop_coalesce(3)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<kBlocksABitSize, false> block_A = 0; block_A < kBlocksA;
           block_A++) {
        for (ac_int<kBlocksBBitSize, false> block_B = 0; block_B < kBlocksB;
             block_B++) {
          for (ac_int<kDrainItersBitSize, false> i = 0; i < kDrainIters; i++) {

            int row_iter = (int)(i) % kRWIterA;
            int col_iter = (int)(i) / kRWIterA;

            // Perform the burst write of pipe_size elements
            fpga_tools::NTuple<T, pipe_size> pipe_read = PipeC::read();
            fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
              int idx = (mat * kMatsizeC) + ((int)(block_B)*tile_B * rows_A) +
                        ((int)(block_A)*tile_A) + (col_iter * rows_A) +
                        (row_iter * pipe_size + k);
              // Only perform the writes that are relevant (and don't access a
              // memory address that may be beyond the matrix last address)
              if constexpr (kIncompleteBurstA) {
                if ((row_iter * pipe_size + k) < tile_A) {
                  BurstCoalescedLSU::store(C_ptr + idx,
                                           pipe_read.template get<k>());
                }
              } else {
                BurstCoalescedLSU::store(C_ptr + idx,
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