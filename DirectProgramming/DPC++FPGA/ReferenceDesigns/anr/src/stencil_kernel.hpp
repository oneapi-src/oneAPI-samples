#ifndef __STENCIL_H__
#define __STENCIL_H__

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "data_bundle.hpp"
#include "shift_reg.hpp"

using namespace sycl;
using namespace hldutils;

namespace hldutils {
  template <typename T>
  constexpr T Min(T a, T b) { return (a < b) ? a : b; }
  template <typename T>
  constexpr T Max(T a, T b) { return (a > b) ? a : b; }
}


// TODO: better way to parameterize WindowFunction so that the user gets nice
// compile-time errors if `WindowFunction` is passed incorrectly. The following
// `using` statement doesn't work because the type is determined by what is
// passed to the templated `stencil` function.

// using StencilFuncType = float(float *stencilArray);

// define a linebuffer like this:
//
//                   ┌─────────────┐
//               ┌───┤ FIFO        ◄───┐
// ┌───┬───┬───┐ │   └─────────────┘   │
// │ r ◄ e ◄ g ◄─┘ ┌───────────────────┘
// ├───┼───┼───┤   │ ┌─────────────┐
// │ r ◄ e ◄ g ◄───┴─┤ FIFO        ◄───┐
// ├───┼───┼───┤     └─────────────┘   │
// │ r ◄ e ◄ g ◄───────────────────────┴─Input
// └───┴───┴───┘
//

// the `WindowFunction` supplied by the user must not have any run-time loops
// in it or it will seriously impact the throughput of this `stencil` task
// function.

using RowType = short;
using ColType = short;

template <typename KernelId, typename StencilType, typename StencilOutputType,
          typename InPipe, typename OutPipe,
          short StencilSize, int MaxImgCols, short ParallelCols,
          typename StencilFunction,
          typename... FunctionArgs_T>
event SubmitStencilKernel(queue& q, RowType rows, ColType cols, const StencilType zeroVal, StencilFunction func, FunctionArgs_T... stencilArgs) {
  using BundledPixels       = DataBundle<StencilType, ParallelCols>;
  using BundledOutputPixels = DataBundle<StencilOutputType, ParallelCols>;

  // TODO: static asserts

  constexpr int ShifterCols = StencilSize + ParallelCols - 1;

  constexpr RowType rowThreshLow = (StencilSize / 2);
  const RowType rowThreshHi      = (StencilSize / 2) + rows;
  constexpr ColType colThreshLow = (StencilSize / 2);
  const ColType colThreshHi      = (StencilSize / 2) + cols;

  constexpr RowType rowOutputThreshLow = 2 * (StencilSize / 2);
  //constexpr ColType colOutputThreshLow = 2 * (StencilSize / 2);

  const ColType fifo_wrap    = (cols + StencilSize - 1 + (ParallelCols - 1 /* round up*/)) / ParallelCols; //(1 + cols + StencilSize) / ParallelCols;

  const RowType paddedRows = rows + 2 * rowThreshLow; // rows + StencilSize - 1
  //const ColType paddedCols = cols + 2 * colThreshLow; // cols + StencilSize - 1

  // separate the loop bound calculation so loop iterations are easier to compute.
  constexpr ColType loop_bound_remainder = ((2 * colThreshLow) / ParallelCols) + (((2 * colThreshLow) % ParallelCols) != 0);

  // Assuming that ParallelCols is a factor of cols, we can streamline the ceiling
  // calculation to a division by a power of 2, and addition of a constant.
  const ColType col_loop_bound = (cols / ParallelCols) + loop_bound_remainder; // ceil(paddedCols/ParallelCols);

  return q.submit([&](handler &h) {
    h.single_task<KernelId>([=] {
      // window shift register
      ShiftReg2d<StencilType, StencilSize, ShifterCols> myShifter;

      // line buffer fifos
      [[intel::fpga_memory]]
      BundledPixels line_buffer_FIFO[(MaxImgCols / ParallelCols) + StencilSize][StencilSize - 1];

      BundledPixels lastNewPixels(zeroVal);
      
      ColType fifoIdx = 0; // track top of FIFO
      //ColType fifoIdx_prev = fifo_wrap; // track top of FIFO

      #if STENCIL_PRELOAD_FIFO
      for (RowType row = 0; row < MaxImgCols + StencilSize; row++) {
          for (ColType col = 0; col < StencilSize - 1; col++) {
              line_buffer_FIFO[row][col] = -1.0f;
          }
      }
      #endif

      [[intel::loop_coalesce(2),
        intel::initiation_interval(1),
        intel::speculated_iterations(0),
        intel::ivdep]]
      for (RowType row = 0; row < paddedRows; row++) {
        [[intel::initiation_interval(1),
          intel::speculated_iterations(0),
          intel::ivdep]]
        for (ColType col_loop = 0; col_loop < col_loop_bound; col_loop++) {
          ColType col = col_loop * ParallelCols;
          BundledPixels newPixels(zeroVal);

          // use shift register and call delegate function
          if ((row >= rowThreshLow) && (row < rowThreshHi) &&
              (col < cols)) {
              newPixels = InPipe::read();
          }

          BundledPixels inputVal(lastNewPixels);
          inputVal.template shiftMultiVals<Min(colThreshLow, ParallelCols), ParallelCols>(newPixels);

          [[intel::fpga_register]]
          BundledPixels pixelColumn[StencilSize];

          // load from FIFO to shift register
          //
          //                   ┌───────────
          // ┌───┬───┬───┐ ┌───┤ FIFO
          // │ r ◄─e ◄─g ◄─┘   └───────────
          // ├───┼───┼───┤     ┌───────────
          // │ r ◄─e ◄─g ◄─────┤ FIFO
          // ├───┼───┼───┤     └───────────
          // │ r ◄─e ◄─g ◄─────────────────Input
          // └───┴───┴───┘

          UnrolledLoop<0, StencilSize>([&](auto stencilRow) {
            if (stencilRow == (StencilSize - 1)) {
                pixelColumn[stencilRow] = inputVal;
            } else {
                pixelColumn[stencilRow] = line_buffer_FIFO[fifoIdx][stencilRow];
            }
          });

          // use this fix: http://www.aerialmantis.co.uk/blog/2017/03/17/template-keywords/
          myShifter.template shiftCols<ParallelCols>(pixelColumn);

          // Continue processing through FIFOs
          //      ┌─────────────┐
          //      │ FIFO        ◄───┐
          //      └─────────────┘   │
          //    ┌───────────────────┘
          //    │ ┌─────────────┐
          //    └─┤ FIFO        ◄───┐
          //      └─────────────┘   │
          //                        └─Input

          UnrolledLoop<0, (StencilSize - 1)>([&](auto fifoRow) {
            if (fifoRow != (StencilSize - 2)) {
                line_buffer_FIFO[fifoIdx][fifoRow] = pixelColumn[fifoRow + 1];
            } else {
                line_buffer_FIFO[fifoIdx][(StencilSize - 2)] = inputVal;
            }
          });

          BundledOutputPixels stencilResults((StencilOutputType)0);

          UnrolledLoop<0, ParallelCols>([&](auto stencilIdx) {
            ShiftReg2d<StencilType, StencilSize, StencilSize> shifterCopy;

            int col_local = col + stencilIdx;

            UnrolledLoop<0, StencilSize>([&](auto stencilRow) {
              UnrolledLoop<0, StencilSize>([&](auto stencilCol) {
                  shifterCopy[stencilRow][stencilCol] =
                    myShifter[stencilRow][stencilCol + stencilIdx];
              });
            });

            // pass a copy of the line buffer's register window.
            StencilOutputType stencilResult = func((row - rowOutputThreshLow), (col_local - ParallelCols), shifterCopy, stencilArgs...);
            stencilResults[stencilIdx]      = stencilResult;
          });

          if ((row >= rowOutputThreshLow) &&
            (col >= colThreshLow) && (col < colThreshHi)) {
            OutPipe::write(stencilResults);
          }

          // increment fifo
          //fifoIdx_prev = fifoIdx;
          fifoIdx++;
          if (fifoIdx == (fifo_wrap)) {
            fifoIdx = (short)0; // Reset Index  /typedef to acint
          }
          lastNewPixels = newPixels;
        }
      }
    });
  });
}

#endif