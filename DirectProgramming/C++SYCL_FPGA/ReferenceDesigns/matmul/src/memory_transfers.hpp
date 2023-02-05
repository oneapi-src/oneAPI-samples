#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

using BurstCoalescedLSU =
    sycl::ext::intel::lsu<sycl::ext::intel::burst_coalesce<true>>;

/**
 * Feeder A Kernel.
 *
 * Reads all "k_num_matrices" matrices from FPGA DDR in bursts of "k_ddr_burst"
 * elements and stores to on-chip memory. Then writes out matrices tile by tile
 * to the pipe, "k_tile_a" elements at a time. Matrices must be provided in
 * column-major order.
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename T,         // Datatype of the elements of the matrix
          int k_rows_a,       // Rows of matrix A
          int k_common,       // Columns of matrix A / rows of matrix B
          int k_cols_b,       // Columns of matrix B
          int k_tile_a,       // Tile size for matrix A
          int k_tile_b,       // Tile size for matrix B
          int k_ddr_burst,    // Number of elements per DDR burst access
          int k_num_matrices, // Number of pairs of matrices to multiply
          typename PipeA>     // Input pipe for matrix
void MatrixReadFromDDRToPipeA(
    T *a_ptr,        // Input matrix pointer
    int repetitions  // Number of times to write the same matrix to the pipe
    ) {

  // May need to perform incomplete memory read if burst size doesn't evenly
  // divide the matrix size
  constexpr bool kIncompleteBurst = k_rows_a % k_ddr_burst != 0;

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;

  // Number of iterations to read from DDR to on-chip memory
  constexpr int kItersPerRowCol
      = k_rows_a / k_ddr_burst + (kIncompleteBurst ? 1 : 0);
  constexpr int kItersToMem = k_common * kItersPerRowCol;
  constexpr int kItersToMemBitSize
      = fpga_tools::BitsForMaxValue<kItersToMem + 1>();

  // Number of iterations to write matrices out to pipe
  constexpr int kItersToPipe = kBlocksA * kBlocksB * k_common;
  constexpr int kItersToPipeBitSize
      = fpga_tools::BitsForMaxValue<kItersToPipe + 1>();

  // Size of a full matrix
  constexpr int kMatsize = k_rows_a * k_common;

  // Memory attributes
  constexpr short kBankWidth = k_ddr_burst * sizeof(T);
  constexpr int kNumBanks
      = fpga_tools::Pow2(fpga_tools::CeilLog2(k_rows_a / k_ddr_burst));

  sycl::device_ptr<T> a_ptr_located(a_ptr);

  // Local memory to store the matrices
  [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
  [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
  [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
  [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
  T mem[k_num_matrices][k_common][k_rows_a];

  // Copy all "k_num_matrices" matrices from FPGA DDR into on-chip memory
  for (int mat = 0; mat < k_num_matrices; mat++) {

    [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
    for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
      int write_idx = i % kItersPerRowCol;

      [[intel::fpga_register]] // NO-FORMAT: Attribute
      T load_reg[k_ddr_burst];

      // Perform the burst read of "k_ddr_burst" elements into register
      // Only perform the reads that are relevant (and don't access a memory
      // address that may be beyond last matrix address)
      fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto k) {
        if ((write_idx * k_ddr_burst + k) < k_rows_a) {
          int ptr_idx = (mat * kMatsize) +
                        (((int)(i) / kItersPerRowCol) * k_rows_a) +
                        (write_idx * k_ddr_burst) + k;
          load_reg[k] = BurstCoalescedLSU::load(a_ptr_located + ptr_idx);
        }
      });

      // Store the "k_ddr_burst" elements into on-chip memory
      fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
        fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto t) {
          if constexpr ((k * k_ddr_burst + t) < k_rows_a) {
            if (write_idx == k) {
              mem[mat][i / kItersPerRowCol][k * k_ddr_burst + t]
                  = load_reg[t];
            }
          }
        });
        write_idx = sycl::ext::intel::fpga_reg(write_idx);
      });
    } // end of i
  }   // end of mat

  // Write every tile of all "k_num_matrices" matrices to the pipe; repeating
  // this operation "repetitions" times to measure performance.
  [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < k_num_matrices; mat++) {

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
        int block = i / (kBlocksB * k_common);

        bool get[kBlocksA];
        fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
          get[k] = block == k;
          block = sycl::ext::intel::fpga_reg(block);
        });

        // Write one column of a matrix tile to the pipe
        fpga_tools::NTuple<T, k_tile_a> pipe_write;
        fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
          fpga_tools::UnrolledLoop<k_tile_a>([&](auto t) {
            pipe_write.template get<t>() =
                get[k] ? mem[mat][i % k_common][k * k_tile_a + t]
                        : sycl::ext::intel::fpga_reg(
                              pipe_write.template get<t>());
          });
        });
        PipeA::write(pipe_write);
      } // end of i
    }   // end of mat
  }     // end of rep
}

/**
 * Feeder B kernel.
 *
 * Reads all "k_num_matrices" matrices from FPGA DDR in bursts of "k_ddr_burst"
 * elements and stores to on-chip memory. Then writes out matrices tile by tile
 * to the pipe, "k_tile_b" elements at a time. Matrices must be provided in
 * row-major order (or, equivalently, given as the transpose).
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename T,         // Datatype of the elements of the matrix
          int k_rows_a,       // Rows of matrix A
          int k_common,       // Columns of matrix A / rows of matrix B
          int k_cols_b,       // Columns of matrix B
          int k_tile_a,       // Tile size for matrix A
          int k_tile_b,       // Tile size for matrix B
          int k_ddr_burst,    // Number of elements per DDR burst access
          int k_num_matrices, // Number of pairs of matrices to multiply
          typename PipeB>     // Input pipe for matrix
void MatrixReadFromDDRToPipeB(
    T *b_ptr,        // Input matrix pointer
    int repetitions  // Number of times to write the same matrix to the pipe
    ) {

  // May need to perform incomplete memory read if burst size doesn't evenly
  // divide the matrix size
  constexpr bool kIncompleteBurst = k_cols_b % k_ddr_burst != 0;

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;

  // Number of iterations to read from DDR to on-chip memory
  constexpr int kItersPerRowCol
      = k_cols_b / k_ddr_burst + (kIncompleteBurst ? 1 : 0);
  constexpr int kItersToMem = k_common * kItersPerRowCol;
  constexpr int kItersToMemBitSize
      = fpga_tools::BitsForMaxValue<kItersToMem + 1>();

  // Number of iterations to write matrices out to pipe
  constexpr int kItersToPipe = kBlocksA * kBlocksB * k_common;
  constexpr int kItersToPipeBitSize
      = fpga_tools::BitsForMaxValue<kItersToPipe + 1>();

  // Size of a full matrix
  constexpr int kMatsize = k_cols_b * k_common;

  // Memory attributes
  constexpr short kBankWidth = k_ddr_burst * sizeof(T);
  constexpr int kNumBanks
      = fpga_tools::Pow2(fpga_tools::CeilLog2(k_cols_b / k_ddr_burst));

  sycl::device_ptr<T> b_ptr_located(b_ptr);

  // Local memory to store the matrices
  [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
  [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
  [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
  [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
  T mem[k_num_matrices][k_common][k_cols_b];

  // Copy all "k_num_matrices" matrices from FPGA DDR into on-chip memory
  for (int mat = 0; mat < k_num_matrices; mat++) {

    [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
    for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
      int write_idx = i % kItersPerRowCol;

      [[intel::fpga_register]] // NO-FORMAT: Attribute
      T load_reg[k_ddr_burst];

      // Perform the burst read of "k_ddr_burst" elements into register
      // Only perform the reads that are relevant (and don't access a memory
      // address that may be beyond last matrix address)
      fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto k) {
        if ((write_idx * k_ddr_burst + k) < k_cols_b) {
          int ptr_idx = (mat * kMatsize) +
                        (((int)(i) / kItersPerRowCol) * k_cols_b) +
                        (write_idx * k_ddr_burst) + k;
          load_reg[k] = BurstCoalescedLSU::load(b_ptr_located + ptr_idx);
        }
      });

      // Store the "k_ddr_burst" elements into on-chip memory
      fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
        fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto t) {
          if constexpr ((k * k_ddr_burst + t) < k_cols_b) {
            if (write_idx == k) {
              mem[mat][i / kItersPerRowCol][k * k_ddr_burst + t]
                  = load_reg[t];
            }
          }
        });
        write_idx = sycl::ext::intel::fpga_reg(write_idx);
      });
    } // end of i
  }   // end of mat

  // Write every tile of all "k_num_matrices" matrices to the pipe; repeating
  // this operation "repetitions" times to measure performance.
  [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < k_num_matrices; mat++) {

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
        int block = (i % (kBlocksB * k_common)) / k_common;

        bool get[kBlocksB];
        fpga_tools::UnrolledLoop<kBlocksB>([&](auto k) {
          get[k] = block == k;
          block = sycl::ext::intel::fpga_reg(block);
        });

        // Write one row of a matrix tile to the pipe
        fpga_tools::NTuple<T, k_tile_b> pipe_write;
        fpga_tools::UnrolledLoop<kBlocksB>([&](auto k) {
          fpga_tools::UnrolledLoop<k_tile_b>([&](auto t) {
            pipe_write.template get<t>() =
                get[k] ? mem[mat][i % k_common][k * k_tile_b + t]
                        : sycl::ext::intel::fpga_reg(
                              pipe_write.template get<t>());
          });
        });
        PipeB::write(pipe_write);
      } // end of i
    }   // end of mat
  }     // end of rep
}

/**
 * Drain Kernel.
 *
 * Reads all "k_num_matrices" matrices tile by tile from the pipe into local
 * memory, "k_tile_a" elements at a time. Then writes the final matrix to FPGA
 * DDR in bursts of "k_ddr_burst" elements. Matrices are stored in column-major
 * order.
 *
 * Repeats this operation "repetitions" times.
 *
 */
template <typename T,         // Datatype of the elements of the matrix
          int k_rows_a,       // Rows of matrix A
          int k_cols_b,       // Columns of matrix B
          int k_tile_a,       // Tile size for matrix A
          int k_tile_b,       // Tile size for matrix B
          int k_ddr_burst,    // Number of elements per DDR burst access
          int k_num_matrices, // Number of pairs of matrices to multiply
          typename PipeC>     // Input pipe for matrix
void MatrixReadPipeToDDR(
    T *c_ptr,        // Input matrix pointer
    int repetitions  // Number of times to write the same matrix to the pipe
    ) {

  // May need to perform incomplete memory read if burst size doesn't evenly
  // divide the tile size
  constexpr bool kIncompleteBurst = k_rows_a % k_ddr_burst != 0;

  // Number of tiles
  constexpr int kBlocksA = k_rows_a / k_tile_a;
  constexpr int kBlocksB = k_cols_b / k_tile_b;

  // Number of iterations to read matrices from pipe
  constexpr int kItersFromPipe = kBlocksA * kBlocksB * k_tile_b;
  constexpr int kItersFromPipeBitSize
      = fpga_tools::BitsForMaxValue<kItersFromPipe + 1>();

  // Number of iterations to write from on-chip memory to DDR
  constexpr int kItersPerRowCol
      = k_rows_a / k_ddr_burst + (kIncompleteBurst ? 1 : 0);
  constexpr int kItersFromMem = k_cols_b * kItersPerRowCol;
  constexpr int kItersFromMemBitSize
      = fpga_tools::BitsForMaxValue<kItersFromMem + 1>();

  // Size of a full matrix
  constexpr int kMatsize = k_rows_a * k_cols_b;

  // Memory attributes
  constexpr short kBankWidth = k_ddr_burst * sizeof(T);
  constexpr int kNumBanks
      = fpga_tools::Pow2(fpga_tools::CeilLog2(k_rows_a / k_ddr_burst));

  sycl::device_ptr<T> c_ptr_located(c_ptr);

  // Local memory to store the matrices
  [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
  [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
  [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
  [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
  T mem[k_num_matrices][k_cols_b][k_rows_a];

  // Read every tile of all "k_num_matrices" matrices from the pipe into
  // on-chip memory; this operation was repeated "repetitions" times to
  // measure performance.
  [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
  for (int rep = 0; rep < repetitions; rep++) {
    for (int mat = 0; mat < k_num_matrices; mat++) {

      [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
      for (ac_int<kItersFromPipeBitSize, false> i = 0; i < kItersFromPipe;
            i++) {
        int block_a = i / (kBlocksB * k_tile_b);
        int block_b = (i % (kBlocksB * k_tile_b)) / k_tile_b;

        // Read one column of a tile of the matrix from the pipe and store to 
        // on-chip memory "mem"
        fpga_tools::NTuple<T, k_tile_a> pipe_read = PipeC::read();
        fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
          fpga_tools::UnrolledLoop<k_tile_a>([&](auto t) {
            if (block_a == k) {
              mem[mat][block_b * k_tile_b + (i % k_tile_b)][k * k_tile_a + t]
                  = pipe_read.template get<t>();
            }
          });
        });
      } // end of i
    }   // end of mat
  }     // end of rep

  // Copy all matrices from on-chip memory "mem" to FPGA DDR
  for (int mat = 0; mat < k_num_matrices; mat++) {
    [[intel::initiation_interval(1)]]   // NO-FORMAT: Attribute
    [[intel::ivdep]]                    // NO-FORMAT: Attribute
    for (ac_int<kItersFromMemBitSize, false> i = 0; i < kItersFromMem; i++) {
      int write_idx = i % kItersPerRowCol;

      bool get[kItersPerRowCol];
      fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
        get[k] = write_idx == k;
        write_idx = sycl::ext::intel::fpga_reg(write_idx);
      });

      [[intel::fpga_register]] // NO-FORMAT: Attribute
      T load_reg[k_ddr_burst];

      // Load "k_ddr_burst" items from on-chip memory into register
      fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
        fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto t) {
          if constexpr ((k * k_ddr_burst + t) < k_rows_a) {
            load_reg[t] =
                get[k] ? mem[mat][i / kItersPerRowCol][k * k_ddr_burst + t]
                        : sycl::ext::intel::fpga_reg(load_reg[t]);
          }
        });
      });

      // Perform the burst write of "k_ddr_burst" elements to DDR
      // Only perform the writes that are relevant (and don't access a memory
      // address that may be beyond last matrix address)
      fpga_tools::UnrolledLoop<k_ddr_burst>([&](auto k) {
        if ((write_idx * k_ddr_burst + k) < k_rows_a) {
          int ptr_idx = (mat * kMatsize) +
                        (((int)(i) / kItersPerRowCol) * k_rows_a) +
                        (write_idx * k_ddr_burst) + k;
          BurstCoalescedLSU::store(c_ptr_located + ptr_idx, load_reg[k]);
        }
      });
    } // end of i
  }   // end of mat
}

#endif /* __MEMORY_TRANSFERS_HPP__ */