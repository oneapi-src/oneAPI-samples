#pragma once
#include <stdint.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "convolution_types.hpp"

//////////////////////////////////////////////////////
// Convert RGB to Grayscale for the convolution
//////////////////////////////////////////////////////
class ID_RGB2Grey;

template <typename PipeIn, typename PipeOut>
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
      conv2d::RGBBeat rgb_beat = PipeIn::read();
      conv2d::GreyScaleBeat grey_beat;

      grey_beat.empty = rgb_beat.empty;
      grey_beat.sop = rgb_beat.sop;
      grey_beat.eop = rgb_beat.eop;

#pragma unroll
      for (int i = 0; i < conv2d::kParallelPixels; i++) {
        grey_beat.data[i] = (rgb_beat.data[i].r / 4) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].g / 2) +  // NO-FORMAT: Alignment
                            (rgb_beat.data[i].b / 4);   // NO-FORMAT: Alignment
      }
      PipeOut::write(grey_beat);
    }
  }
};