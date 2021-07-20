#ifndef __ANR_HPP__
#define __ANR_HPP__

#include <vector>
#include <utility>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "anr_params.hpp"
#include "vertical_kernel.hpp"
#include "horizontal_kernel.hpp"
#include "data_bundle.hpp"
#include "shift_reg.hpp"

using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ONEAPI::experimental::printf(_format, ## __VA_ARGS__); }

// declare the kernel and pipe names globally to reduce name mangling
class IntraPipeID;
class VerticalKernelID;
class HorizontalKernelID;

//
// A struct to carry pixel and intensity sigma data
//
template<typename T>
struct PixelSigIPair {
  PixelSigIPair() {}
  PixelSigIPair(T _pixel) : pixel(_pixel), sig_i(0) {}
  PixelSigIPair(T _pixel, float _sig_i) : pixel(_pixel), sig_i(_sig_i) {}
  T pixel;
  float sig_i;
};

//
// Compute a 1D gaussian (ignores 1/2*pi constant)
//
template<typename T>
constexpr inline T Gaussian1D(T x, T sigma) {
  return sycl::exp(-0.5 * sycl::pown((x / sigma), 2));
}

//
// Build a 1D gaussian filter
//
template<typename T, int size>
constexpr auto BuildGaussianFilter1D(T sigma) {
  ShiftReg<T, size> filter;
  for (int x = -size/2; x <= size/2; x++) {
    filter[x + size/2] = Gaussian1D<T>(x, sigma);
  }
  return filter;
}

//
// Compute the intensity sigma (sig_i) using a pixel value (x) and the ANR
// parameters.
//
float ComputeIntensitySigma(float x, ANRParams params) {
  return sycl::sqrt(params.k * x + params.sig_shot_2) * params.sig_i_coeff;
}

//
// compute intensity sigma give parameters and a pixel value
//
template<typename InT, int filter_size, int filter_size_eff=(filter_size + 1)/2>
inline float BilateralFilter1D(ShiftReg<InT, filter_size>& buffer,
                              ShiftReg<float, filter_size_eff>& spatial_filter,
                              ANRParams params, InT middle_pixel, float sig_i) {
  // static asserts
  static_assert(std::is_arithmetic_v<InT>);
  static_assert(std::is_signed_v<InT>);
  static_assert(filter_size > 1);

  // build the bilateral filter
  ShiftReg<float, filter_size_eff> filter;
  float filter_sum = 0.0;

  UnrolledLoop<0, filter_size_eff>([&](auto i) {
    // TODO: precompute stuff
    // TODO: use a gaussian LUT to improve Gaussian1D DSP usage
    //      Do so for sycl::exp() and sycl::pown()
    // TODO: remove floating point subtraction for vertical version
    const InT intensity_diff = std::abs(middle_pixel - buffer[i*2]);
    const float intensity_gaussian = Gaussian1D<float>(intensity_diff, sig_i);

    // compute the filter value based on the intensity and spatial filters
    const float filter_val = spatial_filter[i] * intensity_gaussian;
    filter[i] = filter_val;
    filter_sum += filter_val;
  });

  // apply the bilateral filter
  float filtered_pixel = 0.0;
  UnrolledLoop<0, filter_size_eff>([&](auto i) {
    filtered_pixel += (buffer[i*2] * filter[i]);
  });
  filtered_pixel /= filter_sum;

  return filtered_pixel; 
}

template<typename InT, typename OutT, int filter_size,
         int filter_size_eff=(filter_size + 1)/2>
struct VerticalFunctor {
  PixelSigIPair<OutT> operator()(int row, int col,
                                 ShiftReg<InT, filter_size> buffer,
                                 ShiftReg<float, filter_size_eff> spatial_filter,
                                 ANRParams params) const {
    // use a short to store the pixels. We need to hold all 8 bits, but 
    // need the type to be signed for the BilateralFilter1D
    using SignedInT = short;

    // static asserts
    static_assert(std::is_arithmetic_v<InT>);
    static_assert(std::is_arithmetic_v<OutT>);
    static_assert(filter_size > 1);
    static_assert(std::is_signed_v<SignedInT>);
    static_assert(sizeof(SignedInT) > sizeof(InT));

    ShiftReg<SignedInT, filter_size> buffer_signed;
    UnrolledLoop<0, filter_size>([&](auto i) {
      buffer_signed[i] = static_cast<SignedInT>(buffer[i]);
    });

    // get the middle index and compute the intensity sigma from it
    constexpr int kMidIdx = filter_size / 2;
    const SignedInT middle_pixel = buffer[kMidIdx];
    const auto sig_i = ComputeIntensitySigma(middle_pixel, params);

    // perform the vertical 1D bilateral filter
    auto res =
      BilateralFilter1D<SignedInT, filter_size>(buffer_signed, spatial_filter,
                                                params, middle_pixel, sig_i);

    // return the result, which is a temporary pixel (res) and the intensity
    // sigma that was calculated for the pixel. This will be forwarded to the
    // horizontal kernel.
    return PixelSigIPair<OutT>(res, sig_i);
  }
};

template<typename InT, typename OutT, int filter_size,
         int filter_size_eff=(filter_size + 1)/2>
struct HorizontalFunctor {
  OutT operator()(int row, int col,
                  ShiftReg<PixelSigIPair<InT>, filter_size> buffer,
                  ShiftReg<float, filter_size_eff> spatial_filter,
                  ANRParams params) const {
    // static asserts
    static_assert(std::is_arithmetic_v<InT>);
    static_assert(std::is_arithmetic_v<OutT>);
    static_assert(filter_size > 1);

    // grab the middle pixel and the intensity sigma for it (forwarded from the
    // vertical kernel)
    constexpr int kMidIdx = filter_size / 2;
    const float middle_pixel = buffer[kMidIdx].pixel;
    const float sig_i = buffer[kMidIdx].sig_i;

    // grab just the pixel data
    ShiftReg<float, filter_size> buffer_pixels;
    UnrolledLoop<0, filter_size>([&](auto i) {
      buffer_pixels[i] = buffer[i].pixel;
    });

    // perform the horizontal 1D bilateral filter
    auto res =
      BilateralFilter1D<float, filter_size>(buffer_pixels, spatial_filter, params,
                                            middle_pixel, sig_i);

    // return the result casted back to a pixel
    // TODO: round rather than cast???
    return OutT(res);
  }
};

//
// Submit all of the ANR kernels (vertical and horizontal)
//
template<typename PixelT, typename IndexT, typename InPipe, 
         typename OutPipe, int filter_size,
         int pixels_per_cycle, int max_cols=4096>
std::vector<event> SubmitANRKernels(queue& q, ANRParams params,
                                    int cols, int rows, int frames) {
  // types
  using VerticalInT = PixelT;
  using VerticalOutT = PixelSigIPair<float>;
  using HorizontalInT = VerticalOutT;
  using HorizontalOutT = PixelT;

  // the internal pipe between the vertical and horizontal kernels
  using IntraPipeT = DataBundle<VerticalOutT, pixels_per_cycle>;
  using IntraPipe = sycl::INTEL::pipe<IntraPipeID, IntraPipeT>;

  // static asserts for template parameters
  static_assert(filter_size > 1);
  static_assert(max_cols > 1);
  static_assert(pixels_per_cycle > 0);
  static_assert(max_cols > pixels_per_cycle);
  static_assert(std::is_arithmetic_v<PixelT>);
  static_assert(std::is_integral_v<IndexT>);

  // validate the arguments
  if (cols > max_cols) {
    std::cerr << "ERROR: cols exceeds the maximum (max_cols)"
              << "(" << cols << " >= " << max_cols << ")\n";
    std::terminate();
  } else if (cols <= 0) {
    std::cerr << "ERROR: cols must be strictly positive\n";
    std::terminate();
  } else if (rows <= 0) {
    std::cerr << "ERROR: rows must be strictly positive\n";
    std::terminate();
  } else if (frames <= 0) {
    std::cerr << "ERROR: frames must be strictly positive\n";
    std::terminate();
  }
  
  // create the filter for the stencil operation
  constexpr int filter_size_eff = (filter_size + 1) / 2; // ceil(filter_size/2)
  auto spatial_filter =
    BuildGaussianFilter1D<float, filter_size_eff>(params.sig_s);

  // the functors for the stencil (vertical) and line (horizontal) kernels
  // either use a functor or a lamda, both work
  auto stencil_func = VerticalFunctor<PixelT, float, filter_size>();
  auto line_func = HorizontalFunctor<float, PixelT, filter_size>();

  // submit the vertical and horizontal kernels, respectively
  auto vertical_kernel =
    SubmitVerticalKernel<VerticalKernelID, VerticalInT, VerticalOutT, IndexT, InPipe, IntraPipe,
                         filter_size, max_cols, pixels_per_cycle>(q, rows, cols, frames, VerticalInT(0), stencil_func, spatial_filter, params);

  auto horizontal_kernel =
    SubmitHorizontalKernel<HorizontalKernelID, HorizontalInT, HorizontalOutT, IndexT, IntraPipe, OutPipe,
                           filter_size, pixels_per_cycle>(q, rows, cols, frames, HorizontalInT(0), line_func, spatial_filter, params);
  
  return {vertical_kernel, horizontal_kernel};
}

#endif  /* __ANR_HPP__ */