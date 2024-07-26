
#pragma once
#include <stdint.h>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "convolution_types.hpp"

//////////////////////////////////////////////////////
// Convert Grayscale to RGB for the display
//////////////////////////////////////////////////////
class ID_Grey2RGB;

template <typename PipeIn, typename PipeOut>
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
      conv2d::GreyScaleBeat grey_beat = PipeIn::read();
      conv2d::RGBBeat rgb_beat;
      rgb_beat.empty = grey_beat.empty;
      rgb_beat.sop = grey_beat.sop;
      rgb_beat.eop = grey_beat.eop;

#pragma unroll
      for (int i = 0; i < conv2d::kParallelPixels; i++) {
        rgb_beat.data[i].r = grey_beat.data[i];  //
        rgb_beat.data[i].g = grey_beat.data[i];  //
        rgb_beat.data[i].b = grey_beat.data[i];  //
      }
      PipeOut::write(rgb_beat);
    }
  }
};