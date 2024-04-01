#ifndef __MEMORY_TRANSFERS_HPP__
#define __MEMORY_TRANSFERS_HPP__

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

/*
  Read matrix_count matrices of type TT from DDR by bursts of num_elem_per_bank
  elements, and write the matrices to the "MatrixPipe" pipe num_elem_per_bank by
  num_elem_per_bank elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Output matrix pipe
          >
void MatrixReadFromDDRToPipeByBlocks(
    TT* matrix_ptr,    // Input matrix pointer
    int matrix_count,  // Number of matrix to read from DDR
    int repetitions    // Number of time to write the same matrix to the pipe
) {
  static_assert(columns % rows == 0,
                "In order to be able to send the matrix by blocs, the number "
                "of rows must be a multiple of the number of columns");

  constexpr int kMatrixSize = rows * columns;
  constexpr int kBlockCount = columns / rows;

  // Repeatedly read matrix_count matrices from DDR and sends them to the pipe
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      for (int block_index = 0; block_index < kBlockCount; block_index++) {
        for (int row = 0; row < rows; row++) {
          for (int column = 0; column < rows; column += num_elem_per_bank) {
            // Read num_elem_per_bank elements per burst
            fpga_tools::NTuple<TT, num_elem_per_bank> ddr_read;
            fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
              if (column + k < rows) {
                ddr_read.template get<k>() =
                    matrix_ptr[matrix_index * kMatrixSize + block_index * rows +
                               row * columns + column + k];
              }
            });

            MatrixPipe::write(ddr_read);
          }  // end of column
        }    // end of row
      }      // end of block_index
    }        // end of matrix_index
  }          // end of repetition
}

/*
  Write matrix_count matrices of type TT from a pipe, num_elem_per_bank by
  num_elem_per_bank and write them to DDR by bursts of num_elem_per_bank
  elements.
  Repeat this operations "repetitions" times.
*/
template <typename TT,            // Datatype of the elements of the matrix
          int rows,               // Number of rows of the matrix
          int columns,            // Number of columns of the matrix
          int num_elem_per_bank,  // Number of TT elements per DDR burst access
          typename MatrixPipe     // Input matrix
          >
void MatrixReadPipeToDDR(
    TT* matrix_ptr,    // Output matrix pointer
    int matrix_count,  // Number of matrix to write to DDR
    int repetitions    // Number of time to read the same matrix to the pipe
) {
  // We may perform an incomplete memory write if the number of elements per row
  // is not a multiple of the DDR burst size
  constexpr bool kIncompleteBurst = rows % num_elem_per_bank != 0;
  constexpr int kExtraIteration = kIncompleteBurst ? 1 : 0;
  // Number of DDR burst of num_elem_per_bank required to write a full column
  constexpr int kLoopIterationsPerColumn =
      rows / num_elem_per_bank + kExtraIteration;
  // Number of DDR burst of num_elem_per_bank to write all the matrices
  constexpr int kLoopIterations = kLoopIterationsPerColumn * columns;
  // Size of a full matrix
  constexpr int kMatrixSize = rows * columns;

  // Repeatedly read matrix_count matrices from the pipe and write them to DDR
  for (int repetition = 0; repetition < repetitions; repetition++) {
    for (int matrix_index = 0; matrix_index < matrix_count; matrix_index++) {
      // Keep track of the current element index in the output matrix
      // Only useful in the case of kIncompleteBurst
      int write_idx = 0;

      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      [[intel::ivdep]]                   // NO-FORMAT: Attribute
      for (int i = 0; i < kLoopIterations; i++) {
        fpga_tools::NTuple<TT, num_elem_per_bank> pipe_read =
            MatrixPipe::read();

        bool last_burst_of_col;
        if constexpr (kIncompleteBurst) {
          // Check if we are writing the last DDR burst of the current column
          last_burst_of_col =
              (i % kLoopIterationsPerColumn) == kLoopIterationsPerColumn - 1;
        }

        fpga_tools::UnrolledLoop<num_elem_per_bank>([&](auto k) {
          if constexpr (kIncompleteBurst) {
            // Check if the current write index is beyond the end of the current
            // matrix column
            bool out_of_bounds =
                last_burst_of_col && (k > ((rows - 1) % num_elem_per_bank));

            // Only perform the DDR writes that are relevant (and don't access a
            // memory address that may be beyond the buffer last address)
            if (!out_of_bounds) {
              matrix_ptr[matrix_index * kMatrixSize + write_idx + k] =
                  pipe_read.template get<k>();
            }
          } else {
            matrix_ptr[matrix_index * kMatrixSize + i * num_elem_per_bank + k] =
                pipe_read.template get<k>();
          }
        });

        if constexpr (kIncompleteBurst) {
          // Update the current element index in the write buffer according
          // to the write size of the current iteration
          write_idx +=
              last_burst_of_col ? rows % num_elem_per_bank : num_elem_per_bank;
        }
      }  // end of i
    }    // end of matrix_index
  }      // end of repetition
}

/*
  Write vector_count vectors of type TT from a pipe, one element at the time and
  write them to DDR. Repeat this operations "repetitions" times.
*/
template <typename TT,         // Datatype of the elements of the matrix
          int size,            // Number of rows of the matrix
          typename VectorPipe  // Input matrix
          >
void VectorReadPipeToDDR(
    TT* vector_ptr,    // Output matrix pointer
    int vector_count,  // Number of vectors to write to DDR
    int repetitions    // Number of time to read the same matrix to the pipe
) {
#if defined(IS_BSP)
  // When targeting a BSP, we instruct the compiler that this pointer
  // lives on the device.
  // Knowing this, the compiler won't generate hardware to
  // potentially get data from the host.
  sycl::device_ptr<TT> vector_ptr_located(vector_ptr);
#else
  // Device pointers are not supported when targeting an FPGA
  // family/part
  TT* vector_ptr_located(vector_ptr);
#endif

  // Repeat vector_count complete R matrix pipe reads
  // for as many repetitions as needed
  for (int repetition = 0; repetition < repetitions; repetition++) {
    [[intel::loop_coalesce(2)]]  // NO-FORMAT: Attribute
    for (int vector_index = 0; vector_index < vector_count; vector_index++) {
      for (int k = 0; k < size; k++) {
        vector_ptr_located[vector_index * size + k] = VectorPipe::read();
      }  // end of k
    }    // end of vector_index
  }      // end of repetition
}

/**
 * Feeder A Kernel.
 *
 * Reads all "num_matrices" matrices from FPGA DDR "elems_per_ddr_access"
 * elements at a time and stores to on-chip memory. Then writes out matrices
 * tile by tile to the pipe, "tile_a" elements at a time. Matrices must be
 * provided in column-major order.
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename TT,              // Datatype of the elements of the matrix
          int aspace,               // Buffer location for mmhost
          int rows_a,               // Rows of matrix A
          int common,               // Columns of matrix A / rows of matrix B
          int cols_b,               // Columns of matrix B
          int tile_a,               // Tile size for matrix A
          int tile_b,               // Tile size for matrix B
          int elems_per_ddr_access, // Number of elements per DDR access
          int num_matrices,         // Number of pairs of matrices to multiply
          typename PipeA,           // Input pipe for matrix
          typename PipeDone, // Pipe to notify compute kernel when to stop
                             // reading inputs
          int datawidth = elems_per_ddr_access * sizeof(TT) * 8>
class MatrixReadFromDDRToPipeA {
public:
#if !defined(IS_BSP)
  // Customizing mmhost only supported when targetting an FPGA part/family
  sycl::ext::oneapi::experimental::annotated_arg<TT *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::awidth<28>,
          sycl::ext::intel::experimental::buffer_location<aspace>,
          sycl::ext::intel::experimental::dwidth<datawidth>,
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::maxburst<1>,
          sycl::ext::intel::experimental::read_write_mode_read,
          sycl::ext::intel::experimental::wait_request_requested})>
#else
  TT *
#endif
      a_ptr;       // Input matrix pointer
  int repetitions; // Number of times to write the same matrix to the pipe

  void operator()() const {
    // May need to perform incomplete memory read if access size doesn't evenly
    // divide the matrix size
    constexpr bool kIncompleteBurst = rows_a % elems_per_ddr_access != 0;
    // Number of tiles
    constexpr int kBlocksA = rows_a / tile_a;
    constexpr int kBlocksB = cols_b / tile_b;
    // Number of iterations to read from DDR to on-chip memory
    constexpr int kItersPerRowCol =
        rows_a / elems_per_ddr_access + (kIncompleteBurst ? 1 : 0);
    constexpr int kItersToMem = common * kItersPerRowCol;
    constexpr int kItersToMemBitSize =
        fpga_tools::BitsForMaxValue<kItersToMem + 1>();
    // Number of iterations to write matrices out to pipe
    constexpr int kItersToPipe = kBlocksA * kBlocksB * common;
    constexpr int kItersToPipeBitSize =
        fpga_tools::BitsForMaxValue<kItersToPipe + 1>();
    // Size of a full matrix
    constexpr int kMatsize = rows_a * common;
    // Memory attributes
    constexpr short kBankWidth = elems_per_ddr_access * sizeof(TT);
    constexpr int kNumBanks =
        fpga_tools::Pow2(fpga_tools::CeilLog2(rows_a / elems_per_ddr_access));

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
    TT mem[num_matrices][common][rows_a];

    // Copy all "num_matrices" matrices from FPGA DDR into on-chip memory
    for (int mat = 0; mat < num_matrices; mat++) {
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
        int write_idx = i % kItersPerRowCol;

        [[intel::fpga_register]] // NO-FORMAT: Attribute
        TT load_reg[elems_per_ddr_access];

        // Perform the read of "elems_per_ddr_access" elements into register
        // Only perform the reads that are relevant (and don't access a memory
        // address that may be beyond last matrix address)
        fpga_tools::UnrolledLoop<elems_per_ddr_access>([&](auto k) {
          if ((write_idx * elems_per_ddr_access + k) < rows_a) {
            int ptr_idx = (mat * kMatsize) +
                          (((int)(i) / kItersPerRowCol) * rows_a) +
                          (write_idx * elems_per_ddr_access) + k;
            load_reg[k] = a_ptr_located[ptr_idx];
          }
        });
        // Store the "elems_per_ddr_access" elements into on-chip memory
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          write_idx = sycl::ext::intel::fpga_reg(write_idx);
          fpga_tools::UnrolledLoop<elems_per_ddr_access>([&](auto t) {
            load_reg[t] = sycl::ext::intel::fpga_reg(load_reg[t]);
            if constexpr ((k * elems_per_ddr_access + t) < rows_a) {
              if (write_idx == k) {
                mem[mat][i / kItersPerRowCol][k * elems_per_ddr_access + t] =
                    load_reg[t];
              }
            }
          });
        });
      } // end of i
    }   // end of mat

    // Write every tile of all "num_matrices" matrices to the pipe; repeating
    // this operation "repetitions" times to measure performance.
    [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
    for (int rep = 0; rep < repetitions; rep++) {
      for (int mat = 0; mat < num_matrices; mat++) {
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
          int block = i / (kBlocksB * common);
          bool get[kBlocksA];
          fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
            block = sycl::ext::intel::fpga_reg(block);
            get[k] = block == k;
          });
          // Write one column of a matrix tile to the pipe
          fpga_tools::NTuple<TT, tile_a> pipe_write;
          fpga_tools::UnrolledLoop<kBlocksA>([&](auto k) {
            fpga_tools::UnrolledLoop<tile_a>([&](auto t) {
              pipe_write.template get<t>() =
                  get[k] ? mem[mat][i % common][k * tile_a + t]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<t>());
            });
          });
          bool last_pipe_write = (rep == repetitions - 1) &
                                 (mat == num_matrices - 1) &
                                 (i == kItersToPipe - 1);
          PipeA::write(pipe_write);
          PipeDone::write(last_pipe_write);
        } // end of i
      }   // end of mat
    }     // end of rep
  }       // end of operator
};

/**
 * Feeder B kernel.
 *
 * Reads all "num_matrices" matrices from FPGA DDR "elems_per_ddr_access"
 * elements at a time and stores to on-chip memory. Then writes out matrices
 * tile by tile to the pipe, "tile_b" elements at a time. Matrices must be
 * provided in row-major order (or, equivalently, given as the transpose).
 *
 * Repeats this operation "repetitions" times to measure performance.
 *
 * Coordinates with the other feeder kernel to support matrix tiling by
 * repeating each tile accordingly.
 *
 */
template <typename TT,              // Datatype of the elements of the matrix
          int aspace,               // Buffer location for mmhost
          int rows_a,               // Rows of matrix A
          int common,               // Columns of matrix A / rows of matrix B
          int cols_b,               // Columns of matrix B
          int tile_a,               // Tile size for matrix A
          int tile_b,               // Tile size for matrix B
          int elems_per_ddr_access, // Number of elements per DDR access
          int num_matrices,         // Number of pairs of matrices to multiply
          typename PipeB,           // Input pipe for matrix
          int datawidth = elems_per_ddr_access * sizeof(TT) * 8>
class MatrixReadFromDDRToPipeB {
public:
#if !defined(IS_BSP)
  // Customizing mmhost only supported when targetting an FPGA part/family
  sycl::ext::oneapi::experimental::annotated_arg<TT *, 
      decltype(sycl::ext::oneapi::experimental::properties{
          sycl::ext::intel::experimental::awidth<28>,
          sycl::ext::intel::experimental::buffer_location<aspace>,
          sycl::ext::intel::experimental::dwidth<datawidth>,
          sycl::ext::intel::experimental::latency<0>,
          sycl::ext::intel::experimental::maxburst<1>,
          sycl::ext::intel::experimental::read_write_mode_read,
          sycl::ext::intel::experimental::wait_request_requested})>
#else
  TT *
#endif
      b_ptr;       // Input matrix pointer
  int repetitions; // Number of times to write the same matrix to the pipe

  void operator()() const {
    // May need to perform incomplete memory read if access size doesn't evenly
    // divide the matrix size
    constexpr bool kIncompleteBurst = cols_b % elems_per_ddr_access != 0;
    // Number of tiles
    constexpr int kBlocksA = rows_a / tile_a;
    constexpr int kBlocksB = cols_b / tile_b;
    // Number of iterations to read from DDR to on-chip memory
    constexpr int kItersPerRowCol =
        cols_b / elems_per_ddr_access + (kIncompleteBurst ? 1 : 0);
    constexpr int kItersToMem = common * kItersPerRowCol;
    constexpr int kItersToMemBitSize =
        fpga_tools::BitsForMaxValue<kItersToMem + 1>();
    // Number of iterations to write matrices out to pipe
    constexpr int kItersToPipe = kBlocksA * kBlocksB * common;
    constexpr int kItersToPipeBitSize =
        fpga_tools::BitsForMaxValue<kItersToPipe + 1>();
    // Size of a full matrix
    constexpr int kMatsize = cols_b * common;
    // Memory attributes
    constexpr short kBankWidth = elems_per_ddr_access * sizeof(TT);
    constexpr int kNumBanks =
        fpga_tools::Pow2(fpga_tools::CeilLog2(cols_b / elems_per_ddr_access));

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
    TT mem[num_matrices][common][cols_b];

    // Copy all "num_matrices" matrices from FPGA DDR into on-chip memory
    for (int mat = 0; mat < num_matrices; mat++) {
      [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
      for (ac_int<kItersToMemBitSize, false> i = 0; i < kItersToMem; i++) {
        int write_idx = i % kItersPerRowCol;

        [[intel::fpga_register]] // NO-FORMAT: Attribute
        TT load_reg[elems_per_ddr_access];

        // Perform the read of "elems_per_ddr_access" elements into register
        // Only perform the reads that are relevant (and don't access a memory
        // address that may be beyond last matrix address)
        fpga_tools::UnrolledLoop<elems_per_ddr_access>([&](auto k) {
          if ((write_idx * elems_per_ddr_access + k) < cols_b) {
            int ptr_idx = (mat * kMatsize) +
                          (((int)(i) / kItersPerRowCol) * cols_b) +
                          (write_idx * elems_per_ddr_access) + k;
            load_reg[k] = b_ptr_located[ptr_idx];
          }
        });
        // Store the "elems_per_ddr_access" elements into on-chip memory
        fpga_tools::UnrolledLoop<kItersPerRowCol>([&](auto k) {
          write_idx = sycl::ext::intel::fpga_reg(write_idx);
          fpga_tools::UnrolledLoop<elems_per_ddr_access>([&](auto t) {
            load_reg[t] = sycl::ext::intel::fpga_reg(load_reg[t]);
            if constexpr ((k * elems_per_ddr_access + t) < cols_b) {
              if (write_idx == k) {
                mem[mat][i / kItersPerRowCol][k * elems_per_ddr_access + t] =
                    load_reg[t];
              }
            }
          });
        });
      } // end of i
    }   // end of mat

    // Write every tile of all "num_matrices" matrices to the pipe; repeating
    // this operation "repetitions" times to measure performance.
    [[intel::loop_coalesce(2)]] // NO-FORMAT: Attribute
    for (int rep = 0; rep < repetitions; rep++) {
      for (int mat = 0; mat < num_matrices; mat++) {
        [[intel::initiation_interval(1)]] // NO-FORMAT: Attribute
        for (ac_int<kItersToPipeBitSize, false> i = 0; i < kItersToPipe; i++) {
          int block = (i % (kBlocksB * common)) / common;
          bool get[kBlocksB];
          fpga_tools::UnrolledLoop<kBlocksB>([&](auto k) {
            block = sycl::ext::intel::fpga_reg(block);
            get[k] = block == k;
          });
          // Write one row of a matrix tile to the pipe
          fpga_tools::NTuple<TT, tile_b> pipe_write;
          fpga_tools::UnrolledLoop<kBlocksB>([&](auto k) {
            fpga_tools::UnrolledLoop<tile_b>([&](auto t) {
              pipe_write.template get<t>() =
                  get[k] ? mem[mat][i % common][k * tile_b + t]
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


#endif /* __MEMORY_TRANSFERS_HPP__ */