#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

/**
 * Feeder A Kernel.
 *
 * Reads all "k_num_matrices" matrices from FPGA DDR "k_elems_per_ddr_access"
 * elements at a time and stores to on-chip memory. Then writes out matrices
 * tile by tile to the pipe, "k_tile_a" elements at a time. Matrices must be
 * provided in column-major order.
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename TT,                // Datatype of the elements of the matrix
          int k_aspace,               // Buffer location for mmhost
          int k_rows_a,               // Rows of matrix A
          int k_common,               // Columns of matrix A / rows of matrix B
          int k_cols_b,               // Columns of matrix B
          int k_tile_a,               // Tile size for matrix A
          int k_tile_b,               // Tile size for matrix B
          int k_elems_per_ddr_access, // Number of elements per DDR access
          int k_num_matrices,         // Number of pairs of matrices to multiply
          typename PipeA,             // Input pipe for matrix
          typename PipeD,
          int k_dwidth = k_elems_per_ddr_access * sizeof(TT) * 8>
class MatrixReadFromDDRToPipeA {
public:
#if !defined(IS_BSP)
  // Customizing mmhost only supported when targetting an FPGA part/family
  mmhost(k_aspace, // buffer_location or aspace
         28,       // address width
         k_dwidth, // data width
         16,       // latency
         1,        // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,        // maxburst
         0,        // align, 0 defaults to alignment of the type
         1)        // waitrequest, 0: false, 1: true
#endif
      TT *a_ptr;   // Input matrix pointer
  int repetitions; // Number of times to write the same matrix to the pipe

  void operator()() const {
    // May need to perform incomplete memory read if access size doesn't evenly
    // divide the matrix size
    constexpr bool kIncompleteBurst = k_rows_a % k_elems_per_ddr_access != 0;
    // Number of tiles
    constexpr int kBlocksA = k_rows_a / k_tile_a;
    constexpr int kBlocksB = k_cols_b / k_tile_b;
    // Number of iterations to read from DDR to on-chip memory
    constexpr int kItersPerRowCol =
        k_rows_a / k_elems_per_ddr_access + (kIncompleteBurst ? 1 : 0);
    constexpr int kItersToMem = k_common * kItersPerRowCol;
    constexpr int kItersToMemBitSize =
        fpga_tools::BitsForMaxValue<kItersToMem + 1>();
    // Number of iterations to write matrices out to pipe
    constexpr int kItersToPipe = kBlocksA * kBlocksB * k_common;
    constexpr int kItersToPipeBitSize =
        fpga_tools::BitsForMaxValue<kItersToPipe + 1>();
    // Size of a full matrix
    constexpr int kMatsize = k_rows_a * k_common;
    // Memory attributes
    constexpr short kBankWidth = k_elems_per_ddr_access * sizeof(TT);
    constexpr int kNumBanks = fpga_tools::Pow2(
        fpga_tools::CeilLog2(k_rows_a / k_elems_per_ddr_access));

#if defined(IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer lives on
    // the device.
    // Knowing this, the compiler won't generate hardware to potentially get
    // data from the host.
    sycl::device_ptr<TT> a_ptr_located(a_ptr);
#else
    // Device pointers are not supported when targeting an FPGA family/part
    TT *a_ptr_located(a_ptr);
#endif

    // Local memory to store the matrices
    [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
    [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
    [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
    TT mem[k_num_matrices][k_common][k_rows_a];

    // Copy all "k_num_matrices" matrices from FPGA DDR into on-chip memory
    for (int mat = 0; mat < k_num_matrices; mat++) {
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
        int write_idx = i % kItersPerRowCol;

        [[intel::fpga_register]] // NO-FORMAT: Attribute
        TT load_reg[k_elems_per_ddr_access];

        // Perform the read of "k_elems_per_ddr_access" elements into register
        // Only perform the reads that are relevant (and don't access
        // a memory address that may be beyond last matrix address)
        fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto k) {
          if ((write_idx * k_elems_per_ddr_access + k) < k_rows_a) {
            int ptr_idx = (mat * kMatsize) +
                          (((int)(i) / kItersPerRowCol) * k_rows_a) +
                          (write_idx * k_elems_per_ddr_access) + k;
            load_reg[k] = a_ptr_located[ptr_idx];
          }
        });
        // Store the "k_elems_per_ddr_access" elements into on-chip memory
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          write_idx = sycl::ext::intel::fpga_reg(write_idx);
          fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto t) {
            load_reg[t] = sycl::ext::intel::fpga_reg(load_reg[t]);
            if constexpr ((k * k_elems_per_ddr_access + t) < k_rows_a) {
              if (write_idx == k) {
                mem[mat][i / kItersPerRowCol][k * k_elems_per_ddr_access + t] =
                    load_reg[t];
              }
            }
          });
        });
      } // end of i
    }   // end of mat

    // Write every tile of all "k_num_matrices" matrices to the pipe; repeating
    // this operation "repetitions" times to measure performance.
    [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
    for (int rep = 0; rep < repetitions; rep++) {
      for (int mat = 0; mat < k_num_matrices; mat++) {
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
          int block = i / (kBlocksB * k_common);
          bool get[kBlocksA];
          fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
            block = sycl::ext::intel::fpga_reg(block);
            get[k] = block == k;
          });
          // Write one column of a matrix tile to the pipe
          fpga_tools::NTuple<TT, k_tile_a> pipe_write;
          fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
            fpga_tools::UnrolledLoop<k_tile_a>([&](auto t) {
              pipe_write.template get<t>() =
                  get[k] ? mem[mat][i % k_common][k * k_tile_a + t]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<t>());
            });
          });
          bool last_pipe_write = (rep == repetitions - 1) &
                                 (mat == k_num_matrices - 1) &
                                 (i == kItersToPipe - 1);
          PipeA::write(pipe_write);
          PipeD::write(last_pipe_write);
        } // end of i
      }   // end of mat
    }     // end of rep
  }       // end of operator
};

/**
 * Feeder B kernel.
 *
 * Reads all "k_num_matrices" matrices from FPGA DDR "k_elems_per_ddr_access"
 * elements at a time and stores to on-chip memory. Then writes out matrices
 * tile by tile to the pipe, "k_tile_b" elements at a time. Matrices must be
 * provided in row-major order (or, equivalently, given as the transpose).
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename TT,                // Datatype of the elements of the matrix
          int k_aspace,               // Buffer location for mmhost
          int k_rows_a,               // Rows of matrix A
          int k_common,               // Columns of matrix A / rows of matrix B
          int k_cols_b,               // Columns of matrix B
          int k_tile_a,               // Tile size for matrix A
          int k_tile_b,               // Tile size for matrix B
          int k_elems_per_ddr_access, // Number of elements per DDR access
          int k_num_matrices,         // Number of pairs of matrices to multiply
          typename PipeB,             // Input pipe for matrix
          int k_dwidth = k_elems_per_ddr_access * sizeof(TT) * 8>
class MatrixReadFromDDRToPipeB {
public:
#if !defined(IS_BSP)
  // Customizing mmhost only supported when targetting an FPGA part/family
  mmhost(k_aspace, // buffer_location or aspace
         28,       // address width
         k_dwidth, // data width
         16,       // latency
         1,        // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,        // maxburst
         0,        // align, 0 defaults to alignment of the type
         1)        // waitrequest, 0: false, 1: true
#endif
      TT *b_ptr;   // Input matrix pointer
  int repetitions; // Number of times to write the same matrix to the pipe

  void operator()() const {
    // May need to perform incomplete memory read if access size doesn't evenly
    // divide the matrix size
    constexpr bool kIncompleteBurst = k_cols_b % k_elems_per_ddr_access != 0;
    // Number of tiles
    constexpr int kBlocksA = k_rows_a / k_tile_a;
    constexpr int kBlocksB = k_cols_b / k_tile_b;
    // Number of iterations to read from DDR to on-chip memory
    constexpr int kItersPerRowCol =
        k_cols_b / k_elems_per_ddr_access + (kIncompleteBurst ? 1 : 0);
    constexpr int kItersToMem = k_common * kItersPerRowCol;
    constexpr int kItersToMemBitSize =
        fpga_tools::BitsForMaxValue<kItersToMem + 1>();
    // Number of iterations to write matrices out to pipe
    constexpr int kItersToPipe = kBlocksA * kBlocksB * k_common;
    constexpr int kItersToPipeBitSize =
        fpga_tools::BitsForMaxValue<kItersToPipe + 1>();
    // Size of a full matrix
    constexpr int kMatsize = k_cols_b * k_common;
    // Memory attributes
    constexpr short kBankWidth = k_elems_per_ddr_access * sizeof(TT);
    constexpr int kNumBanks = fpga_tools::Pow2(
        fpga_tools::CeilLog2(k_cols_b / k_elems_per_ddr_access));

#if defined(IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer lives on
    // the device.
    // Knowing this, the compiler won't generate hardware to potentially get
    // data from the host.
    sycl::device_ptr<TT> b_ptr_located(b_ptr);
#else
    // Device pointers are not supported when targeting an FPGA family/part
    TT *b_ptr_located(b_ptr);
#endif

    // Local memory to store the matrices
    [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
    [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
    [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
    TT mem[k_num_matrices][k_common][k_cols_b];

    // Copy all "k_num_matrices" matrices from FPGA DDR into on-chip memory
    for (int mat = 0; mat < k_num_matrices; mat++) {
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
        int write_idx = i % kItersPerRowCol;

        [[intel::fpga_register]] // NO-FORMAT: Attribute
        TT load_reg[k_elems_per_ddr_access];

        // Perform the read of "k_elems_per_ddr_access" elements into register
        // Only perform the reads that are relevant (and don't access a memory
        // address that may be beyond last matrix address)
        fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto k) {
          if ((write_idx * k_elems_per_ddr_access + k) < k_cols_b) {
            int ptr_idx = (mat * kMatsize) +
                          (((int)(i) / kItersPerRowCol) * k_cols_b) +
                          (write_idx * k_elems_per_ddr_access) + k;
            load_reg[k] = b_ptr_located[ptr_idx];
          }
        });
        // Store the "k_elems_per_ddr_access" elements into on-chip memory
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          write_idx = sycl::ext::intel::fpga_reg(write_idx);
          fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto t) {
            load_reg[t] = sycl::ext::intel::fpga_reg(load_reg[t]);
            if constexpr ((k * k_elems_per_ddr_access + t) < k_cols_b) {
              if (write_idx == k) {
                mem[mat][i / kItersPerRowCol][k * k_elems_per_ddr_access + t] =
                    load_reg[t];
              }
            }
          });
        });
      } // end of i
    }   // end of mat

    // Write every tile of all "k_num_matrices" matrices to the pipe; repeating
    // this operation "repetitions" times to measure performance.
    [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
    for (int rep = 0; rep < repetitions; rep++) {
      for (int mat = 0; mat < k_num_matrices; mat++) {
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
          int block = (i % (kBlocksB * k_common)) / k_common;
          bool get[kBlocksB];
          fpga_tools::UnrolledLoop<kBlocksB>([&](auto k) {
            block = sycl::ext::intel::fpga_reg(block);
            get[k] = block == k;
          });
          // Write one row of a matrix tile to the pipe
          fpga_tools::NTuple<TT, k_tile_b> pipe_write;
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
  }       // end of operator
};

/**
 * Drain Kernel.
 *
 * Reads all "k_num_matrices" matrices tile by tile from the pipe into local
 * memory, "k_tile_a" elements at a time. Then writes the final matrix to FPGA
 * DDR "k_elems_per_ddr_access" elements at a time. Matrices are stored in
 * column-major order.
 *
 * Repeats this operation "repetitions" times.
 *
 */
template <typename TT,                // Datatype of the elements of the matrix
          int k_aspace,               // Buffer location for mmhost
          int k_rows_a,               // Rows of matrix A
          int k_cols_b,               // Columns of matrix B
          int k_tile_a,               // Tile size for matrix A
          int k_tile_b,               // Tile size for matrix B
          int k_elems_per_ddr_access, // Number of elements per DDR access
          int k_num_matrices,         // Number of pairs of matrices to multiply
          typename PipeC,             // Output pipe for matrix
          int k_dwidth = k_elems_per_ddr_access * sizeof(TT) * 8>
class MatrixReadPipeToDDR {
public:
#if !defined(IS_BSP)
  // Customizing mmhost only supported when targetting an FPGA part/family
  mmhost(k_aspace, // buffer_location or aspace
         28,       // address width
         k_dwidth, // data width
         16,       // latency
         2,        // read_write_mode, 0: ReadWrite, 1: Read, 2: Write
         1,        // maxburst
         0,        // align, 0 defaults to alignment of the type
         1)        // waitrequest, 0: false, 1: true
#endif
      TT *c_ptr;   // Output matrix pointer
  int repetitions; // Number of time to read the same matrix to the pipe

  void operator()() const {
    // May need to perform incomplete memory read if DDR access size doesn't
    // evenly divide the tile size
    constexpr bool kIncompleteBurst = k_rows_a % k_elems_per_ddr_access != 0;
    // Number of tiles
    constexpr int kBlocksA = k_rows_a / k_tile_a;
    constexpr int kBlocksB = k_cols_b / k_tile_b;
    // Number of iterations to read matrices from pipe
    constexpr int kItersFromPipe = kBlocksA * kBlocksB * k_tile_b;
    constexpr int kItersFromPipeBitSize =
        fpga_tools::BitsForMaxValue<kItersFromPipe + 1>();
    // Number of iterations to write from on-chip memory to DDR
    constexpr int kItersPerRowCol =
        k_rows_a / k_elems_per_ddr_access + (kIncompleteBurst ? 1 : 0);
    constexpr int kItersFromMem = k_cols_b * kItersPerRowCol;
    constexpr int kItersFromMemBitSize =
        fpga_tools::BitsForMaxValue<kItersFromMem + 1>();
    // Size of a full matrix
    constexpr int kMatsize = k_rows_a * k_cols_b;
    // Memory attributes
    constexpr short kBankWidth = k_elems_per_ddr_access * sizeof(TT);
    constexpr int kNumBanks = fpga_tools::Pow2(
        fpga_tools::CeilLog2(k_rows_a / k_elems_per_ddr_access));

#if defined(IS_BSP)
    // When targeting a BSP, we instruct the compiler that this pointer lives on
    // the device.
    // Knowing this, the compiler won't generate hardware to potentially get
    // data from the host.
    sycl::device_ptr<TT> c_ptr_located(c_ptr);
#else
    // Device pointers are not supported when targeting an FPGA family/part
    TT *c_ptr_located(c_ptr);
#endif

    // Local memory to store the matrices
    [[intel::numbanks(kNumBanks)]]   // NO-FORMAT: Attribute
    [[intel::bankwidth(kBankWidth)]] // NO-FORMAT: Attribute
    [[intel::private_copies(1)]]     // NO-FORMAT: Attribute
    [[intel::max_replicates(1)]]     // NO-FORMAT: Attribute
    TT mem[k_num_matrices][k_cols_b][k_rows_a];

    // Read every tile of all "k_num_matrices" matrices from the pipe into
    // on-chip memory; this operation was repeated "repetitions" times to
    // measure performance.
    [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
    for (int rep = 0; rep < repetitions; rep++) {
      for (int mat = 0; mat < k_num_matrices; mat++) {
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kItersFromPipeBitSize, false> i = 0; i < kItersFromPipe;
             i++) {
          int block_a = i / (kBlocksB * k_tile_b);
          int block_b = (i % (kBlocksB * k_tile_b)) / k_tile_b;
          // Read one column of a tile of the matrix from the pipe and store to
          // on-chip memory "mem"
          fpga_tools::NTuple<TT, k_tile_a> pipe_read = PipeC::read();
          fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
            block_a = sycl::ext::intel::fpga_reg(block_a);
            fpga_tools::UnrolledLoop<k_tile_a>([&](auto t) {
              pipe_read.template get<t>() =
                  sycl::ext::intel::fpga_reg(pipe_read.template get<t>());
              if (block_a == k) {
                mem[mat][block_b * k_tile_b + (i % k_tile_b)]
                   [k * k_tile_a + t] = pipe_read.template get<t>();
              }
            });
          });
        } // end of i
      }   // end of mat
    }     // end of rep

    // Copy all matrices from on-chip memory "mem" to FPGA DDR
    for (int mat = 0; mat < k_num_matrices; mat++) {
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      [[intel::ivdep]]                  // NO-FORMAT: Attribute
      for (ac_int<kItersFromMemBitSize, false> i = 0; i < kItersFromMem; i++) {
        int write_idx = i % kItersPerRowCol;
        bool get[kItersPerRowCol];
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          write_idx = sycl::ext::intel::fpga_reg(write_idx);
          get[k] = write_idx == k;
        });

        [[intel::fpga_register]] // NO-FORMAT: Attribute
        TT load_reg[k_elems_per_ddr_access];

        // Load "k_elems_per_ddr_access" items from on-chip memory into register
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto t) {
            if constexpr ((k * k_elems_per_ddr_access + t) < k_rows_a) {
              load_reg[t] = get[k] ? mem[mat][i / kItersPerRowCol]
                                        [k * k_elems_per_ddr_access + t]
                                   : sycl::ext::intel::fpga_reg(load_reg[t]);
            }
          });
        });
        // Perform the write of "k_elems_per_ddr_access" elements to DDR
        // Only perform the writes that are relevant (and don't access a memory
        // address that may be beyond last matrix address)
        fpga_tools::UnrolledLoop<k_elems_per_ddr_access>([&](auto k) {
          if ((write_idx * k_elems_per_ddr_access + k) < k_rows_a) {
            int ptr_idx = (mat * kMatsize) +
                          (((int)(i) / kItersPerRowCol) * k_rows_a) +
                          (write_idx * k_elems_per_ddr_access) + k;
            c_ptr_located[ptr_idx] = load_reg[k];
          }
        });
      } // end of i
    }   // end of mat
  }     // end of operator
};

#endif /* __MEMORY_TRANSFERS_HPP__ */