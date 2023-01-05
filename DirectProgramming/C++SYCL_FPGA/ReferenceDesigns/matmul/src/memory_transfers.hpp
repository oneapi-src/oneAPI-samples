#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;
using namespace fpga_tools;

using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<true>>;

template <typename T, int rows_A, int common, int cols_B, int tile_A,
          int tile_B, int pipe_size, typename pipe_A>
void feeder_A(T *A, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool incomplete_burst_A = tile_A % pipe_size != 0;

  // Number of pipe reads to read a full column of A and number of columns
  constexpr int rw_iter_A = tile_A / pipe_size + (incomplete_burst_A ? 1 : 0);
  constexpr int rw_iter_A_bitsize = BitsForMaxValue<rw_iter_A + 1>();
  constexpr int common_bitsize = BitsForMaxValue<common + 1>();

  // Number of tiles
  constexpr int blocks_A = rows_A / tile_A;
  constexpr int blocks_B = cols_B / tile_B;
  constexpr int blocks_A_bitsize = BitsForMaxValue<blocks_A + 1>();
  constexpr int blocks_B_bitsize = BitsForMaxValue<blocks_B + 1>();

  // Size of a full matrix A
  constexpr int matsize_A = rows_A * common;

  device_ptr<T> A_ptr(A);

  // Repeatedly read matrix tiles from global memory and send them to the pipe
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::loop_coalesce(4)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<blocks_A_bitsize, false> block_A = 0; block_A < blocks_A;
           block_A++) {
        for (ac_int<blocks_B_bitsize, false> block_B = 0; block_B < blocks_B;
             block_B++) {
          for (ac_int<common_bitsize, false> i = 0; i < common; i++) {
            for (ac_int<rw_iter_A_bitsize, false> j = 0; j < rw_iter_A; j++) {

              // Perform the burst read of pipe_size elements
              NTuple<T, pipe_size> pipe_write;
              UnrolledLoop<pipe_size>([&](auto k) {
                int idx = (mat * matsize_A) + ((int)(block_A)*tile_A) +
                          ((int)(i)*rows_A) + ((int)(j)*pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (incomplete_burst_A) {
                  if ((j * pipe_size + k) < tile_A) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(A_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(A_ptr + idx);
                }
              });

              pipe_A::write(pipe_write);
            }
          }
        }
      }
    }
  }
}

template <typename T, int rows_A, int common, int cols_B, int tile_A,
          int tile_B, int pipe_size, typename pipe_B>
void feeder_B(T *B, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool incomplete_burst_B = tile_B % pipe_size != 0;

  // Number of pipe reads to read a full row of B and number of rows
  constexpr int rw_iter_B = tile_B / pipe_size + (incomplete_burst_B ? 1 : 0);
  constexpr int rw_iter_B_bitsize = BitsForMaxValue<rw_iter_B + 1>();
  constexpr int common_bitsize = BitsForMaxValue<common + 1>();

  // Number of tiles
  constexpr int blocks_A = rows_A / tile_A;
  constexpr int blocks_B = cols_B / tile_B;
  constexpr int blocks_A_bitsize = BitsForMaxValue<blocks_A + 1>();
  constexpr int blocks_B_bitsize = BitsForMaxValue<blocks_B + 1>();

  // Size of a full matrix B
  constexpr int matsize_B = cols_B * common;

  device_ptr<T> B_ptr(B);

  // Repeatedly read matrix tiles from global memory and send them to the pipe
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::loop_coalesce(4)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<blocks_A_bitsize, false> block_A = 0; block_A < blocks_A;
           block_A++) {
        for (ac_int<blocks_B_bitsize, false> block_B = 0; block_B < blocks_B;
             block_B++) {
          for (ac_int<common_bitsize, false> i = 0; i < common; i++) {
            for (ac_int<rw_iter_B_bitsize, false> j = 0; j < rw_iter_B; j++) {

              // Perform the burst read of pipe_size elements
              NTuple<T, pipe_size> pipe_write;
              UnrolledLoop<pipe_size>([&](auto k) {
                int idx = (mat * matsize_B) + ((int)(block_B)*tile_B) +
                          ((int)(i)*cols_B) + ((int)(j)*pipe_size + k);
                // Only perform the reads that are relevant (and don't access a
                // memory address that may be beyond last matrix address)
                if constexpr (incomplete_burst_B) {
                  if ((j * pipe_size + k) < tile_B) {
                    pipe_write.template get<k>() =
                        BurstCoalescedLSU::load(B_ptr + idx);
                  }
                } else {
                  pipe_write.template get<k>() =
                      BurstCoalescedLSU::load(B_ptr + idx);
                }
              });

              pipe_B::write(pipe_write);
            }
          }
        }
      }
    }
  }
}

template <typename T, int rows_A, int cols_B, int tile_A, int tile_B,
          int pipe_size, typename pipe_C>
void drain(T *C, int repetitions, int num_matrices) {

  // May need to perform incomplete memory read if the tile size if not a
  // multiple of the burst size
  constexpr bool incomplete_burst_A = tile_A % pipe_size != 0;

  // Number of pipe reads to read a full column of C
  constexpr int rw_iter_A = tile_A / pipe_size + (incomplete_burst_A ? 1 : 0);

  // Number of tiles
  constexpr int blocks_A = rows_A / tile_A;
  constexpr int blocks_B = cols_B / tile_B;
  constexpr int blocks_A_bitsize = BitsForMaxValue<blocks_A + 1>();
  constexpr int blocks_B_bitsize = BitsForMaxValue<blocks_B + 1>();

  // Iterations to store matrix to global memory
  constexpr int drain_iters = tile_B * rw_iter_A;
  constexpr int drain_iters_bitsize = BitsForMaxValue<drain_iters + 1>();

  // Size of a full matrix C
  constexpr int matsize_C = rows_A * cols_B;

  device_ptr<T> C_ptr(C);

  // Repeatedly read matrix tiles from pipe and write them to global memory
  [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < num_matrices; mat++) {

      [[intel::initiation_interval(1)]]    // NO-FORMAT: Attribute
      [[intel::ivdep]]                     // NO-FORMAT: Attribute
      [[intel::loop_coalesce(3)]]          // NO-FORMAT: Attribute
      [[intel::speculated_iterations(0)]]  // NO-FORMAT: Attribute
      for (ac_int<blocks_A_bitsize, false> block_A = 0; block_A < blocks_A;
           block_A++) {
        for (ac_int<blocks_B_bitsize, false> block_B = 0; block_B < blocks_B;
             block_B++) {
          for (ac_int<drain_iters_bitsize, false> i = 0; i < drain_iters; i++) {
            
            int row_iter = (int)(i) % rw_iter_A;
            int col_iter = (int)(i) / rw_iter_A;

            // Perform the burst write of pipe_size elements
            NTuple<T, pipe_size> pipe_read = pipe_C::read();
            UnrolledLoop<pipe_size>([&](auto k) {
              int idx = (mat * matsize_C) + ((int)(block_B)*tile_B * rows_A) +
                        ((int)(block_A)*tile_A) + (col_iter * rows_A) +
                        (row_iter * pipe_size + k);
              // Only perform the writes that are relevant (and don't access a
              // memory address that may be beyond the matrix last address)
              if constexpr (incomplete_burst_A) {
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