#ifndef _USV_FROM_EIGENS_HPP_
#define _USV_FROM_EIGENS_HPP_

#include "constexpr_math.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"

template <typename T, bool is_complex, int A_rows, int A_cols, int pipe_size,
          typename AIn,     // InputMatrixPipe2,            size A_rows x A_cols
          typename EValIn,  // R matrix from QR iteration,  size 1D - [A_cols]
          typename EVecIn,  // Q accumulated V input,       size A_cols x A_cols
          typename UOut,    // UTempMatrixPipe              size A_rows x A_rows
          typename SOut,    // SMatrixPipe                  size A_rows x A_cols
          typename VOut>    // VMatrixPipe                  size A_cols x A_cols
struct USVFromEigens {
  void operator()() const {
    using TT = std::conditional_t<is_complex, ac_complex<T>, T>;
    // findout if orthogonalazation is needed on the output

    constexpr int kDiagonalSize =
        (A_rows > A_cols) ? A_cols : A_rows;  // min(rows, cols)
    constexpr int kABlockCount = A_rows / A_cols;
    // Copy a matrix from the pipe to a local memory
    // Number of pipe reads of pipe_size required to read a full column
    constexpr int kRExtraIteration = ((A_cols % pipe_size) != 0) ? 1 : 0;
    constexpr int kSExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
    constexpr int kAExtraIteration = ((A_cols % pipe_size) != 0) ? 1 : 0;
    constexpr int kVExtraIteration = ((A_cols % pipe_size) != 0) ? 1 : 0;
    constexpr int kUExtraIteration = ((A_rows % pipe_size) != 0) ? 1 : 0;
    constexpr int kRLoopIterPerColumn = (A_cols / pipe_size) + kRExtraIteration;
    constexpr int kSLoopIterPerColumn = (A_rows / pipe_size) + kSExtraIteration;
    constexpr int kALoopIterPerColumn = (A_cols / pipe_size) + kAExtraIteration;
    constexpr int kVLoopIterPerColumn = (A_cols / pipe_size) + kVExtraIteration;
    constexpr int kULoopIterPerColumn = (A_rows / pipe_size) + kUExtraIteration;
    // Number of pipe reads of pipe_size to read all the matrices
    constexpr int kRLoopIter = kRLoopIterPerColumn * A_cols;
    constexpr int kSLoopIter = kSLoopIterPerColumn * A_cols;
    constexpr int kALoopIter = kALoopIterPerColumn * A_cols;
    constexpr int kVLoopIter = kVLoopIterPerColumn * A_cols;
    constexpr int kULoopIter = kULoopIterPerColumn * A_rows;
    // Size in bits of the loop iterator over kLoopIter iterations
    constexpr int kSLoopIterBitSize =
        fpga_tools::BitsForMaxValue<kSLoopIter + 1>();
    constexpr int kALoopIterBitSize =
        fpga_tools::BitsForMaxValue<kALoopIter + 1>();
    constexpr int kVLoopIterBitSize =
        fpga_tools::BitsForMaxValue<kRLoopIter + 1>();
    constexpr int kULoopIterBitSize =
        fpga_tools::BitsForMaxValue<kULoopIter + 1>();

    constexpr unsigned short kBankwidth = pipe_size * sizeof(TT);
    constexpr unsigned short kSNumBanks = A_rows / pipe_size;
    constexpr unsigned short kANumBanks = A_rows / pipe_size;
    constexpr unsigned short kVNumBanks = A_cols / pipe_size;
    constexpr unsigned short kUNumBanks = A_rows / pipe_size;

    constexpr short kSNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kSNumBanks));
    constexpr short kANumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kANumBanks));
    constexpr short kVNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kVNumBanks));
    constexpr short kUNumBanksNextPow2 =
        fpga_tools::Pow2(fpga_tools::CeilLog2(kUNumBanks));

    while (1) {
      [[intel::numbanks(kSNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT s_result[A_rows][A_cols];

      [[intel::numbanks(kANumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT a_load[A_cols][A_rows];

      [[intel::numbanks(kVNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT v_load[A_cols][A_cols];

      [[intel::numbanks(kUNumBanksNextPow2)]]  // NO-FORMAT: Attribute
      [[intel::bankwidth(kBankwidth)]]        // NO-FORMAT: Attribute
      [[intel::private_copies(4)]]            // NO-FORMAT: Attribute
      [[intel::max_replicates(1)]]            // NO-FORMAT: Attribute
      TT u_result[A_rows][A_rows];

      // load A 
      for (int block = 0; block < kABlockCount; block++) {
        [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
        for (ac_int<kALoopIterBitSize, false> li = 0; li < kALoopIter; li++) {
          fpga_tools::NTuple<TT, pipe_size> pipe_read_a = AIn::read();

          int write_idx_a = li % kALoopIterPerColumn;
          fpga_tools::UnrolledLoop<kALoopIterPerColumn>([&](auto k) {
            fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
              if constexpr (k * pipe_size + t < A_cols) {
                if (write_idx_a == k) {
                  a_load[li / kALoopIterPerColumn]
                        [k * pipe_size + t + block * A_cols]
                         = pipe_read_a.template get<t>();
                }
              }

              // Delay data signals to create a vine-based data distribution
              // to lower signal fanout.
              pipe_read_a.template get<t>() =
                  sycl::ext::intel::fpga_reg(pipe_read_a.template get<t>());
            });

            write_idx_a = sycl::ext::intel::fpga_reg(write_idx_a);
          });
        }
      }

      // read eigenvalues one by one
      for (int k = 0; k < A_cols; k++) {
        s_result[k][k] = sycl::sqrt(EValIn::read());
      }  // end of k

      // process S (sqrt and zero padding)
      #pragma unroll
      for (int r = 0; r < A_rows; r ++){
          #pragma unroll
          for (int c = 0; c < A_cols; c ++) {
            if (r != c)
              s_result[r][c] = (TT)0.0;
        }
      }

      // load V
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kVLoopIterBitSize, false> li = 0; li < kVLoopIter; li++) {
        fpga_tools::NTuple<TT, pipe_size> pipe_read_v = EVecIn::read();
        // pass down V as is
        VOut::write(pipe_read_v);

        int write_idx_v = li % kVLoopIterPerColumn;
        fpga_tools::UnrolledLoop<kVLoopIterPerColumn>([&](auto k) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto t) {
            if constexpr (k * pipe_size + t < A_cols) {
              if (write_idx_v == k) {
                v_load[k * pipe_size + t][li / kVLoopIterPerColumn] =
                    pipe_read_v.template get<t>();
              }
            }

            // Delay data signals to create a vine-based data distribution
            // to lower signal fanout.
            pipe_read_v.template get<t>() =
                sycl::ext::intel::fpga_reg(pipe_read_v.template get<t>());
          });

          write_idx_v = sycl::ext::intel::fpga_reg(write_idx_v);
        });
      }

      // Compute the matrix product A @ V / S[c][c]
      for (int row = 0; row < A_rows; row++) {
        for (int column = 0; column < A_rows; column++) {
          if (column < kDiagonalSize) {
            TT dot_prod{0};
            fpga_tools::UnrolledLoop<A_cols>([&](auto k) {
              dot_prod = sycl::ext::intel::fpga_reg(dot_prod) +
                         a_load[k][row] * v_load[k][column];
            });

            TT s_val = s_result[column][column];
            u_result[row][column] = dot_prod / s_val;
          } else
            u_result[row][column] = (TT)(row / column);  // filler data
        } // col
      }  // row

      // output s_result
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kSLoopIterBitSize, false> li = 0; li < kSLoopIter; li++) {
        int column_iter = li % kSLoopIterPerColumn;
        bool get[kSLoopIterPerColumn];
        fpga_tools::UnrolledLoop<kSLoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<TT, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kSLoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < A_rows) {
              pipe_write.template get<k>() =
                  get[t] ? s_result[t * pipe_size + k][li / kSLoopIterPerColumn]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        SOut::write(pipe_write);
      }

      // output u_result
      [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
      for (ac_int<kULoopIterBitSize, false> li = 0; li < kULoopIter; li++) {
        int column_iter = li % kULoopIterPerColumn;
        bool get[kULoopIterPerColumn];
        fpga_tools::UnrolledLoop<kULoopIterPerColumn>([&](auto k) {
          get[k] = column_iter == k;
          column_iter = sycl::ext::intel::fpga_reg(column_iter);
        });

        fpga_tools::NTuple<TT, pipe_size> pipe_write;
        fpga_tools::UnrolledLoop<kULoopIterPerColumn>([&](auto t) {
          fpga_tools::UnrolledLoop<pipe_size>([&](auto k) {
            if constexpr (t * pipe_size + k < A_rows) {
              pipe_write.template get<k>() =
                  get[t] ? u_result[t * pipe_size + k][li / kULoopIterPerColumn]
                         : sycl::ext::intel::fpga_reg(
                               pipe_write.template get<k>());
            }
          });
        });
        UOut::write(pipe_write);
      }
    }  // while(1)
  }
};  // struct USVFromEigens 

#endif  // _USV_FROM_EIGENS_HPP_