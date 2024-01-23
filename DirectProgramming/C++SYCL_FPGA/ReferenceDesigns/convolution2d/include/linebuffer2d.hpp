#pragma once

#include "comparisons.hpp"
#include "data_bundle.hpp"
#include "shift_reg.hpp"

namespace linebuffer2d {

#pragma pack(push, 1)
// struct type to wrap a pixel type with sop and eop signals. Could use the
// StreamingBeat type but this results in compilation failures due to
// https://hsdes.intel.com/appstore/article/#/18034529956.
template <typename PixelTypeIn>
struct PixelWithSignals_Templated {
  PixelTypeIn val;
  bool sop;
  bool eop;
  int empty;
};
#pragma pack(pop)

template <typename PixelTypeIn, typename PixelTypeOut, short kStencilSize,
          short kMaxImgCols, short kParallelPixels>
class LineBuffer2d {
 public:
  // types used by LineBuffer2d
  using StencilDataBundleIn =
      fpga_tools::DataBundle<PixelTypeIn, kParallelPixels>;
  using StencilDataBundleOut =
      fpga_tools::DataBundle<PixelTypeOut, kParallelPixels>;

  // public members
  [[intel::fpga_register]]  //
  short rows;

  [[intel::fpga_register]]  //
  short cols;

 private:
  // types used internally
  using PixelWithSignals = PixelWithSignals_Templated<PixelTypeIn>;
  using BundledPixels = fpga_tools::DataBundle<PixelWithSignals, kParallelPixels>;
  constexpr static short kRowWriteInit = (short)(0 - kStencilSize);
  constexpr static short kColWriteInit = (short)(0 - kStencilSize);

  ///////////////////////////////
  // initialize state variables
  ///////////////////////////////
  // infer parameterization of FIFO and register windows
  constexpr static int ShifterCols = kStencilSize + kParallelPixels - 1;

  // line buffer
  [[intel::fpga_register]]  //
  fpga_tools::ShiftReg2d<PixelWithSignals, kStencilSize, ShifterCols>
      myShifter;

  constexpr static int kFifoCols =
      (kMaxImgCols / kParallelPixels) + kStencilSize;
  constexpr static int kFifoRows = kStencilSize - 1;

  // Line buffer for image Data
  [[intel::fpga_memory]]  //
  BundledPixels line_buffer_FIFO[kFifoCols][kFifoRows];

  short fifoIdx = 0;  // track top of FIFO
  short fifo_wrap = cols / kParallelPixels;

  // If we have multiple pixels in parallel, we need to insert some dummy
  // pixels before the 'real data' so the output has the same alignment as
  // the input.
  constexpr static short kBufferOffset = min(
      (short)((kParallelPixels - (short)(kStencilSize / 2))), kParallelPixels);

  constexpr static short kPreBufferSize = kParallelPixels + kBufferOffset;

  [[intel::fpga_register]]  //
  fpga_tools::DataBundle<PixelWithSignals, kPreBufferSize>
      preBuffer;

  // separate the loop bound calculation so loop iterations are easier to
  // compute.
  const short col_loop_bound = (cols / kParallelPixels);

  short row_write = kRowWriteInit;
  short col_loop = 0, col_write = kColWriteInit;
  bool eop_curr = false;
  bool sop_curr = false;
  int empty = 0, empty_curr = 0;

 public:
  LineBuffer2d(short m_rows, short m_cols) : rows(m_rows), cols(m_cols) {}

  /// @brief Callback function used by `filter` function. This function must
  /// accept the following parameters:
  /// @param[in] rows The total number of rows in the image being processed
  /// @param[in] cols The total number of columns in the image being processed
  /// @param[in] row The row of the pixel at the centre of `pixels`.
  /// @param[in] col The column of the pixel at the centre of `pixels`.
  /// @param[in] pixels An array containing the pixels in the window. The centre
  /// of the window corresponds to location (`row`, `col`) in the image.
  template <typename... FunctionArgs_T>
  using WindowFunction = PixelTypeOut (*)(
      short rows, short cols, short row, short col,
      fpga_tools::ShiftReg2d<PixelTypeIn, kStencilSize, kStencilSize> pixels,
      FunctionArgs_T... stencilArgs);

  /// @brief This function structures a sliding-window stencil function in a way
  /// that is optimal for FPGAs. `filter()` inserts a new pixel into the
  /// linebuffer, and runs the user-provided window function. Any additional
  /// arguments you provide this function will be passed to your window function
  /// (in addition to the mandatory arguments it must accept). The return value
  /// of `filter()` is the concatenation of the results of all the copies of
  /// your window function.
  ///
  /// @paragraph Schematic
  ///```                                           <br/>
  ///                   ┌─────────────┐            <br/>
  ///               ┌───┤ FIFO        ◄───┐        <br/>
  /// ┌───┬───┬───┐ │   └─────────────┘   │        <br/>
  /// │ r ◄ e ◄ g ◄─┘ ┌───────────────────┘        <br/>
  /// ├───┼───┼───┤   │ ┌─────────────┐            <br/>
  /// │ r ◄ e ◄ g ◄───┴─┤ FIFO        ◄───┐        <br/>
  /// ├───┼───┼───┤     └─────────────┘   │        <br/>
  /// │ r ◄ e ◄ g ◄───────────────────────┴─Input  <br/>
  /// └───┴───┴───┘                                <br/>
  ///```                                           <br/>
  /// @tparam windowFunction operation to perform on window. Must be a template
  /// parameter rather than a function pointer due to SYCL.
  /// @param[in] newPixels input data
  /// @param[in] isNewFrame Set this to `true` if the pixel(s) you pass in
  /// `newPixels` is/are at the start of a frame.
  /// @param[in] isLineEnd Set this to `true` if the pixel(s) you pass in
  /// `newPixels` is/are at the end of a line.
  /// @param[out] startOfFrame This is set to `true` if the returned pixel(s)
  /// is/are at the start of a new frame.
  /// @param[out] endOfLine This is set to `true` if the returned pixel(s)
  /// is/are at the end of a line.
  /// @return filter result
  template <auto(&windowFunction), typename... FunctionArgs_T>
  StencilDataBundleOut filter(StencilDataBundleIn newPixels, bool isNewFrame,
                              bool isLineEnd, bool &startOfFrame,
                              bool &endOfLine, FunctionArgs_T... stencilArgs) {
    [[intel::fpga_register]]  //
    BundledPixels newPixelsStructs;

#pragma unroll
    for (int i = 0; i < kParallelPixels; i++) {
      // wrap each pixel value in a struct
      PixelTypeIn newPixel = newPixels[i];

      [[intel::fpga_register]]  //
      PixelWithSignals pixelStruct{newPixel, isNewFrame, isLineEnd, empty};
      newPixelsStructs[i] = pixelStruct;
    }

    preBuffer.template shiftMultiVals<kParallelPixels>(newPixelsStructs);

    // grab the first `kParallelPixels` samples to push into the stencil
    [[intel::fpga_register]]  //
    BundledPixels inputVal;
    inputVal.template shiftMultiVals<kParallelPixels, kPreBufferSize>(
        preBuffer);

    [[intel::fpga_register]]  //
    BundledPixels pixelColumn[kStencilSize];

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

    fpga_tools::UnrolledLoop<kStencilSize>([&](auto stencilRow) {
      if constexpr (stencilRow == (kStencilSize - 1)) {
        pixelColumn[stencilRow] = inputVal;
      } else {
        pixelColumn[stencilRow] = line_buffer_FIFO[fifoIdx][stencilRow];
      }
    });

    // use this fix:
    // http://www.aerialmantis.co.uk/blog/2017/03/17/template-keywords/
    myShifter.template shiftCols<kParallelPixels>(pixelColumn);

    // Continue processing through FIFOs
    //    ┌─────────────┐
    //    │ FIFO        ◄───┐
    //    └─────────────┘   │
    //  ┌───────────────────┘
    //  │ ┌─────────────┐
    //  └─┤ FIFO        ◄───┐
    //    └─────────────┘   │
    //                      └─Input
    fpga_tools::UnrolledLoop<(kStencilSize - 1)>([&](auto fifoRow) {
      if constexpr (fifoRow != (kStencilSize - 2)) {
        line_buffer_FIFO[fifoIdx][fifoRow] = pixelColumn[fifoRow + 1];
      } else {
        line_buffer_FIFO[fifoIdx][(kStencilSize - 2)] = inputVal;
      }
    });

    // get the sop and eop signals from the pixel currently being processed
    constexpr int kStencilCenter = (kStencilSize - 1) / 2;

    sop_curr = myShifter[kStencilCenter][kStencilCenter].sop;
    eop_curr = myShifter[kStencilCenter][kStencilCenter].eop;
    empty_curr = myShifter[kStencilCenter][kStencilCenter].empty;

    // compute EOP/SOP by looking at the pixels that will be combined into the
    // output
    /*
        sop_curr = false;
        eop_curr = false;
        empty_curr = kParallelPixels * sizeof(PixelTypeIn);
    #pragma unroll
        for (int i = 0; i < kParallelPixels; i++) {
          sop_curr |= myShifter[kStencilCenter][kStencilCenter + i].sop;
          eop_curr |= myShifter[kStencilCenter][kStencilCenter + i].eop;
          empty_curr = 0;
        }
        */

    // SOP=1 corresponds with the current pixel having Row=0 and col=0.
    // EOP=1 corresponds with the NEXT pixel having row=row+1 and col=0;
    if (sop_curr) {
      row_write = 0;
      col_write = 0;
    }

    StencilDataBundleOut stencilResults;

#pragma unroll
    for (int stencilIdx = 0; stencilIdx < kParallelPixels; stencilIdx++) {
      PixelTypeIn shifterCopy[kStencilSize * kStencilSize];

      short col_local = (col_write + stencilIdx);

      // use the pixels in myShifter to generate a window to pass to
      // windowFunction. This is where we could add some customization to
      // change the 'edge' behaviour (e.g. what do we do when we are at column
      // 0). That is probably up to the user to decide though, so we just copy
      // pixels from myShifter and let windowFunction decide how to handle
      // edges.

#pragma unroll
      for (int stencilRow = 0; stencilRow < kStencilSize; stencilRow++) {
#pragma unroll
        for (int stencilCol = 0; stencilCol < kStencilSize; stencilCol++) {
          shifterCopy[stencilCol + stencilRow * kStencilSize] =
              myShifter[stencilRow][stencilCol + stencilIdx].val;
        }
      }

      // in-line this function on a copy of the appropriate shifter data
      PixelTypeOut stencilResult = windowFunction(
          row_write, col_local, rows, cols, shifterCopy, stencilArgs...);
      stencilResults[stencilIdx] = stencilResult;
    }

    if ((row_write >= 0) && (col_write >= 0)) {
      startOfFrame = sop_curr;
      endOfLine = eop_curr;
    } else {
      startOfFrame = true;
      endOfLine = true;
    }

    // increment fifo fifoIdx_prev = fifoIdx;
    fifoIdx++;
    if (fifoIdx == (fifo_wrap)) {
      // TODO: make the index reset depend on previous EOP read from input to
      // remove need to compare with fifo_wrap.
      fifoIdx = (short)0;  // Reset Index
    }

    // update loop counter variables
    col_loop++;
    if (col_loop == col_loop_bound) {
      col_loop = 0;
    }

    // SOP=1 corresponds with the current pixel having Row=0 and col=0.
    // EOP=1 corresponds with the NEXT pixel having row=row+1 and col=0;
    // if (eop_curr) {
    //   col_write = 0;
    //   row_write++;
    // }

    col_write += kParallelPixels;
    // reset col_write and row_write when SOP and EOP appear

    if (col_write >= cols) {
      col_write = 0;
      row_write++;
    }

    return stencilResults;
  }
};
}  // namespace linebuffer2d