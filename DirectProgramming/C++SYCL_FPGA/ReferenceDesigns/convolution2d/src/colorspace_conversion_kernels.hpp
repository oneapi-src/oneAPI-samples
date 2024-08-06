#pragma once
#include <stdint.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "gray_pixels.hpp"
#include "pipe_matching.hpp"
#include "rgb_pixels.hpp"

//////////////////////////////////////////////////////
// Convert Grayscale to RGB for the display
//////////////////////////////////////////////////////
class ID_Grey2RGB;

/// @brief Convert RGB pixels to grayscale. The pixel input and pixel output
/// pipes are implemented as template parameters so that you can choose how to
/// connect this kernel:
///
///  * Connect this kernel to host pipes to operate it in isolation
///
///  * Connect this kernel to other kernels via inter-kernel pipes to operate it
/// as part of a system.
/// @tparam PipeIn Pipe to read input data from.The payload of this pipe must
/// be a `GrayScaleBeat` as defined in `gray_pixels.hpp`.
/// @tparam PipeOut Pipe to write output data to. The payload of this pipe must
/// be an `RGBBeat` as defined in `rgb_pixels.hpp`.
template <vvp_gray::GrayScalePipe PipeIn, vvp_rgb::RGBPipe PipeOut>
struct Grey2RGB {
  // Kernel properties method to configure the kernel to be a kernel with
  // streaming invocation interface.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_remove_downstream_stall};
  }
  void operator()() const {
    // this loop is not necessary in hardware since the `start` bit will be
    // pulled high, but it makes the testbench easier.
    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (1) {
      vvp_gray::GrayScaleBeat grey_beat = PipeIn::read();
      vvp_rgb::RGBBeat rgb_beat;
      rgb_beat.empty = grey_beat.empty;
      rgb_beat.sop = grey_beat.sop;
      rgb_beat.eop = grey_beat.eop;

#pragma unroll
      for (int i = 0; i < vvp_gray::kParallelPixels; i++) {
        rgb_beat.data[i].r = grey_beat.data[i];  //
        rgb_beat.data[i].g = grey_beat.data[i];  //
        rgb_beat.data[i].b = grey_beat.data[i];  //
      }
      PipeOut::write(rgb_beat);
    }
  }
};

//////////////////////////////////////////////////////
// Convert RGB to Grayscale for the convolution
//////////////////////////////////////////////////////
class ID_RGB2Grey;

/// @brief Convert RGB pixels to grayscale. The pixel input and pixel output
/// pipes are implemented as template parameters so that you can choose how to
/// connect this kernel:
///
///  * Connect this kernel to host pipes to operate it in isolation
///
///  * Connect this kernel to other kernels via inter-kernel pipes to operate it
/// as part of a system.
/// @tparam PipeIn Pipe to read input data from. The payload of this pipe must
/// be an `RGBBeat` as defined in `rgb_pixels.hpp`.
/// @tparam PipeOut Pipe to write output data to. The payload of this pipe must
/// be a `GrayScaleBeat` as defined in `gray_pixels.hpp`.
template <vvp_rgb::RGBPipe PipeIn, vvp_gray::GrayScalePipe PipeOut>
struct RGB2Grey {
  // Kernel properties method to configure the kernel to be a kernel with
  // streaming invocation interface.
  auto get(sycl::ext::oneapi::experimental::properties_tag) {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::
            streaming_interface_remove_downstream_stall};
  }
  void operator()() const {
    // this loop is not necessary in hardware since I will pin the
    // `start` bit high, but it makes the testbench easier.
    [[intel::initiation_interval(1)]]  // NO-FORMAT: Attribute
    while (1) {
      vvp_rgb::RGBBeat rgb_beat = PipeIn::read();
      vvp_gray::GrayScaleBeat grey_beat;

      grey_beat.empty = rgb_beat.empty;
      grey_beat.sop = rgb_beat.sop;
      grey_beat.eop = rgb_beat.eop;

#pragma unroll
      for (int i = 0; i < vvp_gray::kParallelPixels; i++) {
        grey_beat.data[i] = (rgb_beat.data[i].r / 4) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].g / 2) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].b / 4);   // NO-FORMAT: Alignment
      }
      PipeOut::write(grey_beat);
    }
  }
};