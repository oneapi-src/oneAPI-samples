#ifndef __ANR_HPP__
#define __ANR_HPP__

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "anr_params.hpp"

using namespace sycl;

// declare kernel names globally to reduce name mangling
class ANRKernelID;

template<typename PixelT, typename InPipe, typename OutPipe, int filter_size,
         int max_width=4096, int max_height=4096, int pixels_per_cycle_k=1>
std::vector<event> SubmitANRKernels(queue& q, ANRParams params, int w, int h,
                                    int frames) {
  // static asserts for template parameters
  static_assert(filter_size > 0);
  static_assert(max_width > 0);
  static_assert(max_height > 0);
  static_assert(pixels_per_cycle_k > 0);

  // validate the image size
  if (w >= max_width) {
    std::cerr << "ERROR: width ('w') exceeds the maximum width (max_width)"
              << "(" << w << " >= " << max_width << ")\n";
    std::terminate();
  }
  if (h >= max_height) {
    std::cerr << "ERROR: height ('h') exceeds the maximum height (max_height)"
              << "(" << h << " >= " << max_height << ")\n";
    std::terminate();
  }

  // the total number of pixels for a single frame
  const auto frame_pixel_count = w * h;

  // submit the ANR kernels
  auto e = q.submit([&](handler &h) {
    h.single_task<ANRKernelID>([=] {
      // TODO: implement this
      for (int f = 0; f < frames; f++) {
        for (int i = 0; i < frame_pixel_count; i++) {
          auto d = InPipe::read();
          OutPipe::write(d);
        }
      }
    });
  });

  return {e};
}

#endif  /* __ANR_HPP__ */