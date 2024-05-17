//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// vvp_stream_adapters.hpp

#pragma once
#include <algorithm>
#include <string>

// oneAPI headers
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "bmp_tools.hpp"  // BmpTools::PixelRGB definition

// C++ magic that lets us extract template parameters from SYCL pipes,
// `StreamingBeat` structs
#include "extract_typename.hpp"

namespace vvp_stream_adapters {

/// @brief Write a frame to the pipe `PixelPipe` and generate appropriate
/// SoP/EoP and Empty sideband signals
/// @paragraph This function writes the contents of an array of pixels into a
/// sycl pipe that can be consumed by a oneAPI kernel. It generates
/// start-of-packet and end-of-packet sideband signals like a video/vision
/// processing (VVP) FPGA IP would, so you can test that your IP complies with
/// the VVP standard.
/// @tparam PixelPipe The pipe to which pixels will be written. This pipe's
/// payload should be a `StreamingBeat` templated on a `std::array`, which is
/// itself templated on a payload of type `PixelType`.
/// @tparam PixelType The type that represents each pixel. This may be a scalar
/// (such as an `int`) or a `struct` of 'plain old data'.
/// @param q SYCL queue where your oneAPI kernel will run
/// @param[in] rows Image rows (height)
/// @param[in] cols Image columns (width)
/// @param[in] in_img Pointer to a buffer containing a single image to pass to
/// your oneAPI kernel.
/// @param[in] end_pixel Optional parameter that lets you simulate a defective
/// video frame, by ending the stream of pixels prematurely.
/// @return `true` after successfully writing the input image to a SYCL pipe.
template <typename PixelPipe, typename PixelType>
bool WriteFrameToPipe(sycl::queue q, int rows, int cols, PixelType *in_img,
                      int end_pixel = -1) {
  if (end_pixel == -1) end_pixel = rows * cols;

  ///////////////////////////////////////
  // Extract parameters from PixelPipe
  ///////////////////////////////////////

  // the payload of PixelPipe should be a StreamingBeat
  using StreamingBeatType = typename ExtractPipeType<PixelPipe>::value_type;

  // the payload of PixelPipe should be a StreamingBeat, whose payload is a
  // std::array
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel = std::size(DataBundleType{});
  using PixelTypeCalc = typename DataBundleType::value_type;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value,
                "(Pipe Payload, output memory) mismatched");
  if (0 != (cols % kPixelsInParallel)) {
    std::cerr
        << "ERROR: WriteFrameToPipe(): kPixelsInParallel must be a factor "
           "of cols!!";
    return false;
  }

  ///////////////////////////////////////////////
  // Package the pixels in in_img into PixelPipe
  ///////////////////////////////////////////////

  std::cout << "INFO: WriteFrameToPipe(): writing " << end_pixel
            << " pixels to pipe with " << kPixelsInParallel
            << " pixels in parallel. " << std::endl;

  for (int i_base = 0; i_base < end_pixel; i_base += kPixelsInParallel) {
    DataBundleType in_bundle = {};
    // sop at the beginning of each frame
    bool sop = (i_base == 0);
    // eop at the end of each line
    bool eop = (i_base != 0) && (0 == ((i_base + kPixelsInParallel) % (cols)));
    int empty = 0;

    // construct beat with n>=1 parallel pixels
    for (int i_subpixel = 0; i_subpixel < kPixelsInParallel; i_subpixel++) {
      int i = i_base + i_subpixel;
      PixelType subpixel;  // TODO: figure out what to do with structs
      if (i < (rows * cols)) {
        subpixel = in_img[i];
        in_bundle[i_subpixel] = subpixel;
      } else {
        empty++;
      }
    }

    // handle different combinations of usePackets and useEmpty
    if constexpr (BeatUseEmpty<PixelPipe>() && BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle, sop, eop, empty);
      PixelPipe::write(q, in_beat);
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle, sop, eop);
      PixelPipe::write(q, in_beat);
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         !BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle);
      PixelPipe::write(q, in_beat);
    } else {
      std::cerr << "ERROR: WriteFrameToPipe(): Invalid beat parameterization."
                << std::endl;
      return false;
    }
  }
  return true;
}

/// @brief Parse an image from a SYCL pipe into a buffer. This function stores
/// the data read from a SYCL pipe into a buffer pointed to by the parameter
/// `out_img`. If this function detects an un-expected start-of-packet signal,
/// it will print a note and write the new frame over the previous partial
/// frame. It will return once it has read a complete frame, so if your design
/// does not completely output a frame, the `ReadFrameFromPipe()` function will
/// hang.
/// @tparam PixelPipe The pipe from which pixels will be read. This pipe's
/// payload should be a `StreamingBeat` templated on a `std::array`, which is
/// itself templated on a payload of type `PixelType`.
/// @tparam PixelType The type that represents each pixel. This may be a scalar
/// (such as an `int`) or a `struct` of 'plain old data'.
/// @param q SYCL queue where your oneAPI kernel will run
/// @param[in] rows Image rows (height)
/// @param[in] cols Image columns (width)
/// @param[out] out_img Pointer to place image pixels read from `PixelPipe`
/// @param[out] sidebands_ok Indicates if sideband signals are correct
/// @param[out] parsed_frames Indicates how many frames were parsed. In the
/// absence of defective frames, this should be 1.
/// @param[in] print_debug_info If set to true, this function will print
/// information about the consumed pipe data to help with debugging.
/// @return `false` if the packet reader ends in an undefined state
template <typename PixelPipe, typename PixelType>
bool ReadFrameFromPipe(sycl::queue q, int rows, int cols, PixelType *out_img,
                       bool &sidebands_ok, int &parsed_frames,
                       bool print_debug_info = false) {
  ///////////////////////////////////////
  // Extract parameters from PixelPipe
  ///////////////////////////////////////

  // the payload of PixelPipe should be a StreamingBeat
  using StreamingBeatType = typename ExtractPipeType<PixelPipe>::value_type;

  // the payload of PixelPipe should be a StreamingBeat, whose payload is a
  // std::array
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel = std::size(DataBundleType{});
  using PixelTypeCalc = typename DataBundleType::value_type;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value,
                "(Pipe Payload, output memory) mismatched");
  if (0 != (cols % kPixelsInParallel)) {
    std::cerr << "ERROR: ReadFrameFromPipe(): kPixelsInParallel must be a "
                 "factor of cols!!";
    return false;
  }

  assert(cols > kPixelsInParallel &&
         "cols must be larger than kPixelsInParallel so that we never "
         "see SOP=1 and EOP=1 in an image.");

  assert((0 == (cols % kPixelsInParallel)) &&
         "cols must be a multiple of kPixelsInParallel!");

  ////////////////////////////////////////////////////////////
  // Consume the beats from PixelPipe, and place into out_img
  ////////////////////////////////////////////////////////////
  std::cout << "INFO: ReadFrameFromPipe(): reading data from pipe with "
            << kPixelsInParallel << " pixels in parallel. " << std::endl;

  parsed_frames = 0;

  int i_base = 0;
  int eop_count = 0;
  bool passed = true;
  bool saw_sop = false;
  sidebands_ok = true;

  bool is_dummy_beat = true;
  bool was_dummy_beat = true;
  long dummy_beats = 0;

  while (eop_count < rows) {
    // expect SOP at the beginning of each frame
    bool sop_expected = (i_base == 0);
    // expect EOP at the end of each line. This calculation is valid because we
    // require `cols` to be a multiple of `kPixelsInParallel`.
    bool eop_expected = (0 == ((i_base + kPixelsInParallel) % cols));

    bool sop_calc = false;
    bool eop_calc = false;

    // don't check empty since it's not used

    StreamingBeatType out_beat = PixelPipe::read(q);

    // handle different combinations of usePackets and useEmpty
    DataBundleType out_bundle;
    if constexpr (BeatUseEmpty<PixelPipe>() && BeatUsePackets<PixelPipe>()) {
      sop_calc = out_beat.sop;
      eop_calc = out_beat.eop;
      out_bundle = out_beat.data;
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         BeatUsePackets<PixelPipe>()) {
      sop_calc = out_beat.sop;
      eop_calc = out_beat.eop;
      out_bundle = out_beat.data;
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         !BeatUsePackets<PixelPipe>()) {
      sop_calc = sop_expected;
      eop_calc = eop_expected;
      out_bundle = out_beat.data;
    } else {
      std::cerr << "ERROR: ReadFrameFromPipe(): Invalid beat parameterization."
                << std::endl;
      return false;
    }
    if (saw_sop) {
      sidebands_ok = (sop_calc == sop_expected) && (eop_calc == eop_expected);
    }
    if (!sidebands_ok) {
      passed = false;
    }

    if (print_debug_info | !sidebands_ok) {
      char buf[256];
      snprintf(buf, 256,
               "%s: ReadFrameFromPipe(): [i = %d] - expect sop=%s eop=%s. "
               "saw sop=%s eop=%s.",
               (sidebands_ok ? "INFO" : "DEFECT"), i_base,
               (sop_expected ? "TRUE" : "FALSE"),
               (eop_expected ? "TRUE" : "FALSE"), (sop_calc ? "TRUE" : "FALSE"),
               (eop_calc ? "TRUE" : "FALSE"));

      std::cout << buf << std::endl;
    }

    // reset if there is a start-of-packet and keep reading.
    if (sop_calc) {
      parsed_frames++;

      std ::cout << "INFO: ReadFrameFromPipe(): saw start of packet; reset "
                    "counters."
                 << std::endl;

      i_base = 0;
      eop_count = 0;
      passed = true;
      saw_sop = true;
    }

    was_dummy_beat = is_dummy_beat;
    is_dummy_beat = !saw_sop;
    if (is_dummy_beat) dummy_beats++;

    // print info about most recently parsed block of dummy beats
    if (!is_dummy_beat && was_dummy_beat) {
      std::cout << "INFO: ReadFrameFromPipe(): saw a block of " << dummy_beats
                << " dummy beats. "
                   "Counters reset."
                << std::endl;
      dummy_beats = 0;
    }

    for (int i_subpixel = 0; i_subpixel < kPixelsInParallel; i_subpixel++) {
      int i = i_base + i_subpixel;
      if (i < (rows * cols)) {
        PixelType subpixel = out_bundle[i_subpixel];
        out_img[i] = subpixel;
      }
    }

    if (saw_sop) {
      if (eop_calc) eop_count++;

      i_base += kPixelsInParallel;
    }
  }
  std::cout << "INFO: ReadFrameFromPipe(): wrote " << eop_count << " lines. "
            << std::endl;
  return passed;
}

/// @brief Write some dummy pixels to the pipe `PixelPipe` to flush a pipelined
/// kernel. Dummy pixels have both the `start-of-frame` and `end-of-line`
/// signals high, so they will be easily identifiable in simulation waveforms.
/// @paragraph This function writes dummy values into a
/// SYCL* pipe that can be consumed by a oneAPI kernel.
/// @tparam PixelPipe The pipe to which pixels will be written. This pipe's
/// payload should be a `StreamingBeat` templated on a `std::array`, which is
/// itself templated on a payload of type `PixelType`.
/// @tparam PixelType The type that represents each pixel. This may be a scalar
/// (such as an `int`) or a `struct` of 'plain old data'.
/// @param q SYCL queue where your oneAPI kernel will run
/// @param[in] len number of dummy pixels
/// @param[in] val the dummy value to write
/// @return `true` after successfully writing the input image to a SYCL pipe.
template <typename PixelPipe, typename PixelType>
bool WriteDummyPixelsToPipe(sycl::queue q, int len, PixelType val) {
  ///////////////////////////////////////
  // Extract parameters from PixelPipe
  ///////////////////////////////////////

  // the payload of PixelPipe should be a StreamingBeat
  using StreamingBeatType = typename ExtractPipeType<PixelPipe>::value_type;

  // the payload of PixelPipe should be a StreamingBeat, whose payload is a
  // std::array
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel = std::size(DataBundleType{});
  using PixelTypeCalc = typename DataBundleType::value_type;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value,
                "(Pipe Payload, output memory) mismatched");

  ////////////////////////////
  // Package the dummy values
  ////////////////////////////
  std::cout
      << "INFO: WriteDummyPixelsToPipe(): storing dummy pixels to pipe with "
      << kPixelsInParallel << " pixels in parallel. " << std::endl;

  int written_dummy_beats = 0;
  for (int i_base = 0; i_base < len; i_base += kPixelsInParallel) {
    DataBundleType in_bundle = {};
    int empty = 0;
    bool sop = false;
    bool eop = false;

    // construct beat with n>=1 parallel pixels
    for (int i_subpixel = 0; i_subpixel < kPixelsInParallel; i_subpixel++) {
      PixelType subpixel = val;
      in_bundle[i_subpixel] = subpixel;
    }

    // handle different combinations of usePackets and useEmpty
    if constexpr (BeatUseEmpty<PixelPipe>() && BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle, sop, eop, empty);
      PixelPipe::write(q, in_beat);
      written_dummy_beats++;
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle, sop, eop);
      PixelPipe::write(q, in_beat);
      written_dummy_beats++;
    } else if constexpr (!BeatUseEmpty<PixelPipe>() &&
                         !BeatUsePackets<PixelPipe>()) {
      StreamingBeatType in_beat(in_bundle);
      PixelPipe::write(q, in_beat);
      written_dummy_beats++;
    } else {
      std::cerr
          << "ERROR: WriteDummyPixelsToPipe(): Invalid beat parameterization."
          << std::endl;
      return false;
    }
  }
  std::cout << "INFO: WriteDummyPixelsToPipe(): wrote " << written_dummy_beats
            << " dummy streaming beats." << std::endl;
  return true;
}
}  // namespace vvp_stream_adapters