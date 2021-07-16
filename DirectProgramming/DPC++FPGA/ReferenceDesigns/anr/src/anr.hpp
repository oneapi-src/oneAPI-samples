#ifndef __ANR_HPP__
#define __ANR_HPP__

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "anr_params.hpp"
#include "stencil_kernel.hpp"
#include "data_bundle.hpp"
#include "shift_reg.hpp"

using namespace sycl;

// declare kernel names globally to reduce name mangling
class IntraPipeID;
class VerticalKernelID;
class HorizontalKernelID;

template<typename T, int size>
using GenericWindowType = hldutils::ShiftReg2d<T, size, size>;

//
// Compute a 1D gaussian (ignores 1/2*pi constant)
//
template<typename T>
constexpr inline T Gaussian1D(T x, T sigma) {
  // TODO: use sycl::recip?
  return sycl::exp(-0.5 * sycl::pown((x / sigma), 2));
}

//
// Build a 1D gaussian filter
//
template<typename T, int size>
constexpr auto BuildGaussian1DFilter(T sigma) {
  ShiftReg<T, size> filter;
  for (int x = -size/2; x <= size/2; x++) {
    filter[x + size/2] =  Gaussian1D<T>(x, sigma);
  }
  return filter;
}

//
// The functor for the stencil kernel. This performs the vertical filtering.
//
template<typename T, int filter_size, int filter_size_eff=(filter_size + 1)/2>
struct StencilFunctor {
  T operator()(int row, int col,
               GenericWindowType<T, filter_size> buffer,
               ShiftReg<float, filter_size_eff> spatial_filter,
               ANRParams params) const {
    constexpr int mid_idx = filter_size / 2;

    // get the middle pixel and compute sigma_i for it.
    // TODO: precompute some of this?
    const T middle_pixel = buffer[mid_idx][mid_idx];
    const float sig_i =
        sycl::sqrt(params.k * middle_pixel + params.sig_shot_2) * params.sig_i_coeff;

    // compute total filter using the intensity values
    ShiftReg<float, filter_size_eff> filter;
    float filter_sum = 0.0;
    #pragma unroll
    for (int y = 0; y < filter_size_eff; y++) {
      // TODO: precompute stuff
      // TODO: use a gaussian LUT
      const auto intensity_diff = std::abs(middle_pixel - buffer[y*2][mid_idx]);
      const float intensity_gaussian = Gaussian1D<float>(intensity_diff, sig_i);

      // compute the filter value based on the intensity and spatial filters
      const float filter_val = spatial_filter[y] * intensity_gaussian;
      filter[y] = filter_val;
      filter_sum += filter_val;
    }

    // convolve the filter with the pixel window to get the output pixel
    float filtered_pixel = 0.0;
    #pragma unroll
    for (int y = 0; y < filter_size_eff; y++) {
      filtered_pixel += buffer[y*2][mid_idx] * filter[y];
    }
    filtered_pixel /= filter_sum;

    // TODO: round back to unsigned char?
    return T(filtered_pixel);
  }
};

//
// Kernel to perform the horizontal filtering
//
template<typename KernelId, typename PixelT, typename InPipe, typename OutPipe,
         int filter_size, int pixels_per_cycle=1,
         int filter_size_eff=(filter_size + 1)/2>
event SubmitHorizontalKernel(queue& q, int cols, int rows, int frames,
                             ANRParams params,
                             ShiftReg<float, filter_size_eff> spatial_filter) {
  //constexpr kShiftRegSize = filter_size + pixels_per_cycle - 1;
  const auto iterations = (rows * cols * frames) / pixels_per_cycle;

  return q.submit([&](handler &rows) {
    rows.single_task<KernelId>([=] {
      // the shift register
      //ShiftReg<PixelT, kShiftRegSize> shifty;

      for (int i = 0; i < iterations; i++) {
        // read new values from the pipe
        auto d = InPipe::read();
        OutPipe::write(d);
      }
    });
  });
}

//
// Submit all of the ANR kernels (vertical and horizontal)
//
template<typename PixelT, typename InPipe, typename OutPipe, int filter_size,
         int max_cols=4096, int pixels_per_cycle=1>
std::vector<event> SubmitANRKernels(queue& q, ANRParams params,
                                    int cols, int rows, int frames) {
  // static asserts for template parameters
  static_assert(filter_size > 0);
  static_assert(max_cols > 0);
  static_assert(pixels_per_cycle > 0);

  using PipeType = DataBundle<PixelT, pixels_per_cycle>;
  using IntraPipe = sycl::INTEL::pipe<IntraPipeID, PipeType>;

  // validate the image size
  if (cols > max_cols) {
    std::cerr << "ERROR: cols exceeds the maximum (max_cols)"
              << "(" << cols << " >= " << max_cols << ")\n";
    std::terminate();
  }
  
  // create the filter for the stencil operation
  constexpr int filter_size_eff = (filter_size + 1) / 2; // ceil(filter_size/2)
  auto spatial_filter =
    BuildGaussian1DFilter<float, filter_size_eff>(params.sig_s);

  // either use a functor or a lamda, both work
  auto stencil_func = StencilFunctor<PixelT, filter_size>();

  // submit the vertical kernel (stencil kernel)
  auto vertical_kernel =
    SubmitStencilKernel<VerticalKernelID, PixelT, PixelT, InPipe, IntraPipe,
                        filter_size, max_cols, pixels_per_cycle>(q, rows, cols, 0, stencil_func, spatial_filter, params);

  // submit the horizontal kernel
  auto horizontal_kernel =
    SubmitHorizontalKernel<HorizontalKernelID, PixelT, IntraPipe, OutPipe,
                           filter_size, pixels_per_cycle>(q, rows, cols, frames,
                                                          params,
                                                          spatial_filter);

  // TODO: alpha blending?

  return {vertical_kernel, horizontal_kernel};
}

#endif  /* __ANR_HPP__ */