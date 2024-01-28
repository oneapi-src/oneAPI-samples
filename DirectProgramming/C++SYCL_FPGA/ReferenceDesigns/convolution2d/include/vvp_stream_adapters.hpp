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

// allows multiple parallel pixels
#include "data_bundle.hpp"

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
/// payload should be a `StreamingBeat` templated on a `DataBundle`, which is
/// itself templated on a payload of type `PixelType`.
/// @tparam PixelType The type that represents each pixel. This may be a scalar
/// (such as an `int`) or a `struct` of 'plain old data'.
/// @param q SYCL queue where your oneAPI kernel will run
/// @param[in] rows Image rows (height)
/// @param[in] cols Image columns (width)
/// @param in_img Pointer to a buffer containing a single image to pass to your
/// oneAPI kernel.
/// @param end_pixel Optional parameter that lets you simulate a defective video
/// frame, by ending the stream of pixels prematurely.
/// @return `true` after successfully writing the input image to a SYCL pipe.
template <typename PixelPipe, typename PixelType>
bool WriteFrameToPipe(sycl::queue q, int rows, int cols, PixelType *in_img,
                      int end_pixel = -1) {
  if (end_pixel == -1) end_pixel = rows * cols;

  if (end_pixel != rows * cols)
    std::cout << "INFO: WriteFrameToPipe: will end frame early, after "
              << end_pixel << " pixels." << std::endl;

  ///////////////////////////////////////
  // Extract parameters from PixelPipe
  ///////////////////////////////////////

  // the payload of PixelPipe should be a StreamingBeat
  using StreamingBeatType = typename ExtractPipeType<PixelPipe>::value_type;

  // the payload of PixelPipe should be a StreamingBeat, whose payload is a
  // DataBundle
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel =
      fpga_tools::ExtractDataBundleType<DataBundleType>::kBundlePayloadCount;
  using PixelTypeCalc = typename fpga_tools::ExtractDataBundleType<
      DataBundleType>::BundlePayloadT;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value, "mismatched");
  if (0 != (cols % kPixelsInParallel)) {
    std::cerr << "ERROR: kPixelsInParallel must be a factor of cols!!";
    return false;
  }

  ///////////////////////////////////////////////
  // Package the pixels in in_img into PixelPipe
  ///////////////////////////////////////////////

  std::cout << "INFO: Storing data to pipe with " << kPixelsInParallel
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
      std::cerr << "ERROR: Invalid beat parameterization." << std::endl;
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
/// payload should be a `StreamingBeat` templated on a `DataBundle`, which is
/// itself templated on a payload of type `PixelType`.
/// @tparam PixelType The type that represents each pixel. This may be a scalar
/// (such as an `int`) or a `struct` of 'plain old data'.
/// @param q SYCL queue where your oneAPI kernel will run
/// @param[in] rows Image rows (height)
/// @param[in] cols Image columns (width)
/// @param[out] out_img Pointer to place image pixels read from `PixelPipe`
/// @param[out] sidebands_ok Indicates if sideband signals are correct
/// @param[out] defective_frames Indicates how many defective frames were read
/// before the current frame
/// @return `false` if the packet reader ends in an undefined state
template <typename PixelPipe, typename PixelType>
bool ReadFrameFromPipe(sycl::queue q, int rows, int cols, PixelType *out_img,
                       bool &sidebands_ok, int &defective_frames) {
  ///////////////////////////////////////
  // Extract parameters from PixelPipe
  ///////////////////////////////////////

  // the payload of PixelPipe should be a StreamingBeat
  using StreamingBeatType = typename ExtractPipeType<PixelPipe>::value_type;

  // the payload of PixelPipe should be a StreamingBeat, whose payload is a
  // DataBundle
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel =
      fpga_tools::ExtractDataBundleType<DataBundleType>::kBundlePayloadCount;
  using PixelTypeCalc = typename fpga_tools::ExtractDataBundleType<
      DataBundleType>::BundlePayloadT;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value, "mismatched");
  if (0 != (cols % kPixelsInParallel)) {
    std::cerr << "ERROR: kPixelsInParallel must be a factor of cols!!";
    return false;
  }

  ////////////////////////////////////////////////////////////
  // Consume the beats from PixelPipe, and place into out_img
  ////////////////////////////////////////////////////////////

  std::cout << "INFO: Reading data from pipe with " << kPixelsInParallel
            << " pixels in parallel. " << std::endl;

  int eop_count = 0;
  bool passed = true;
  int i_base = 0;

  while (eop_count < rows) {
    // expect SOP at the beginning of each frame
    bool sop_expected = (i_base == 0);
    // expect EOP at the end of each line. This calculation is valid because we
    // require `cols` to be a multiple of `kPixelsInParallel`.
    bool eop_expected = (0 == ((i_base + kPixelsInParallel) % cols));
    int empty_expected =
        (eop_expected
             ? ((kPixelsInParallel - ((rows * cols) % kPixelsInParallel)) %
                kPixelsInParallel)
             : 0);
    bool sop_calc = false;
    bool eop_calc = false;
    int empty_calc = 0;
    StreamingBeatType out_beat = PixelPipe::read(q);
    DataBundleType out_bundle;

    // handle different combinations of usePackets and useEmpty
    if constexpr (BeatUseEmpty<PixelPipe>() && BeatUsePackets<PixelPipe>()) {
      sop_calc = out_beat.sop;
      eop_calc = out_beat.eop;
      empty_calc = out_beat.empty;
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
      std::cerr << "ERROR: Invalid beat parameterization." << std::endl;
      return false;
    }

    // reset if there is an error condition and keep reading.
    if (sop_calc & !sop_expected) {
      std::cout << "INFO: Saw unexpected start of packet; reset counters."
                << std::endl;
      i_base = 0;
      eop_count = 0;
      passed = true;
    }

    for (int i_subpixel = 0; i_subpixel < kPixelsInParallel; i_subpixel++) {
      int i = i_base + i_subpixel;
      if (i < (rows * cols)) {
        PixelType subpixel = out_bundle[i_subpixel];
        out_img[i] = subpixel;
      }
    }
    bool empty_itr = (empty_calc == empty_expected);
    if (!empty_itr) {
      char buf[256];
      snprintf(buf, 256,
               "INFO: ReadFrameFromPipe(): [i = %d] - expect empty=%d. saw "
               "empty=%d.\n",
               i_base, empty_expected, empty_calc);
      std::cout << buf << std::endl;
    }

    bool sop_eop_itr =
        ((sop_calc == sop_expected) && (eop_calc == eop_expected));
    if (!sop_eop_itr) {
      char buf[256];
      snprintf(buf, 256,
               "INFO: ReadFrameFromPipe(): [i = %d] - expect sop=%s eop=%s. "
               "saw sop=%s "
               "eop=%s.",
               i_base, (sop_expected ? "TRUE" : "FALSE"),
               (eop_expected ? "TRUE" : "FALSE"), (sop_calc ? "TRUE" : "FALSE"),
               (eop_calc ? "TRUE" : "FALSE"));
      std::cout << buf << std::endl;
    }
    sidebands_ok = sop_eop_itr & empty_itr;
    if (!sidebands_ok) {
      passed = false;
    }

    if (eop_calc) eop_count++;

    i_base += kPixelsInParallel;
  }
  return passed;
}

/// @brief Write some dummy pixels to the pipe `PixelPipe` to flush a pipelined
/// kernel. Dummy pixels have both the `start-of-frame` and `end-of-line`
/// signals high, so they will be easily identifiable in simulation waveforms.
/// @paragraph This function writes dummy values into a
/// SYCL* pipe that can be consumed by a oneAPI kernel.
/// @tparam PixelPipe The pipe to which pixels will be written. This pipe's
/// payload should be a `StreamingBeat` templated on a `DataBundle`, which is
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
  // DataBundle
  using DataBundleType = BeatPayload<PixelPipe>;
  constexpr int kPixelsInParallel =
      fpga_tools::ExtractDataBundleType<DataBundleType>::kBundlePayloadCount;
  using PixelTypeCalc = typename fpga_tools::ExtractDataBundleType<
      DataBundleType>::BundlePayloadT;

  // sanity check
  static_assert(std::is_same<PixelTypeCalc, PixelType>::value, "mismatched");

  ////////////////////////////
  // Package the dummy values
  ////////////////////////////

  std::cout << "INFO: Storing dummy pixels to pipe with " << kPixelsInParallel
            << " pixels in parallel. " << std::endl;

  int written_dummy_beats = 0;
  for (int i_base = 0; i_base < len; i_base += kPixelsInParallel) {
    DataBundleType in_bundle = {};
    int empty = 0;
    bool sop = true;
    bool eop = true;

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
      std::cerr << "ERROR: Invalid beat parameterization." << std::endl;
      return false;
    }
  }

  std::cout << "Info: Wrote " << written_dummy_beats
            << " dummy streaming beats." << std::endl;
  return true;
}
}  // namespace vvp_stream_adapters