#ifndef __ANR_HPP__
#define __ANR_HPP__

#include <vector>
#include <utility>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "anr_params.hpp"
#include "column_stencil.hpp"
#include "row_stencil.hpp"
#include "data_bundle.hpp"
#include "shift_reg.hpp"
#include "mp_math.hpp"

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
// A struct to carry the new pixel, the original pixel, and the intensity sigma
// from the vertical kernel to horizontal kernel
//
template<typename PixelT>
struct DataForwardStruct {
  DataForwardStruct() {}
  DataForwardStruct(float pixel_n) : pixel_n(pixel_n) {}
  DataForwardStruct(float pixel_n, PixelT pixel_o, float sig_i) :
    pixel_n(pixel_n), pixel_o(pixel_o), sig_i(sig_i) {}
  
  float pixel_n; // the new pixel
  PixelT pixel_o; // the original pixel
  float sig_i; // the sigma 
};

//
// Compute a 1D gaussian (ignores 1/2*pi constant)
//
constexpr inline float Gaussian1D(float x, float sigma) {
  return sycl::exp(-0.5 * sycl::pown((x / sigma), 2));
}

//
// Build a 1D gaussian filter
//
template<int size>
constexpr auto BuildGaussianFilter1D(float sigma) {
  ShiftReg<float, size> filter;
  for (int x = -size/2; x <= size/2; x++) {
    filter[x + size/2] = Gaussian1D(x, sigma);
  }
  return filter;
}

//
// Compute the intensity sigma (sig_i) using a pixel value (x) and the ANR
// parameters.
// TODO: use a LUT for this
//
float ComputeIntensitySigma(float x, ANRParams params) {
  return sycl::sqrt(params.k * x + params.sig_shot_2) * params.sig_i_coeff;
}

//
// Computes a generic 1D bilateral filter.
//
template<typename InT, int filter_size, int filter_size_eff=(filter_size + 1)/2>
inline float BilateralFilter1D(ShiftReg<InT, filter_size>& buffer,
                              ShiftReg<float, filter_size_eff>& spatial_filter,
                              ANRParams params, float sig_i) {
  // static asserts
  static_assert(std::is_arithmetic_v<InT>);
  static_assert(std::is_signed_v<InT>);
  static_assert(filter_size > 1);

  // the middle pixel index
  constexpr int kMidIdx = filter_size/2;

  // build the bilateral filter
  ShiftReg<float, filter_size_eff> filter;
  float filter_sum = 0.0;

  UnrolledLoop<0, filter_size_eff>([&](auto i) {
    // TODO: use a gaussian LUT to improve Gaussian1D DSP usage
    //      Do so for sycl::exp(), sycl::pown()
    const InT intensity_diff = std::abs(buffer[kMidIdx] - buffer[i*2]);
    const float intensity_gaussian = Gaussian1D(intensity_diff, sig_i);

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

//
// Functor for the column stencil callback.
// This performs the 1D vertical bilateral filter. It also computes the
// intensity sigma value (sig_i) and bundles it with the pixel to be forwarded
// to the horizontal kernel.
//
template<typename InT, typename OutT, int filter_size,
         int filter_size_eff=(filter_size + 1)/2>
struct VerticalFunctor {
  auto operator()(int row, int col,
                  ShiftReg<InT, filter_size> buffer,
                  ShiftReg<float, filter_size_eff> spatial_filter,
                  ANRParams params) const {
    // use a short to store the pixels. We need to hold all 8 bits, but 
    // need the type to be signed for the BilateralFilter1D
    using SignedInT = short;

    // static asserts
    static_assert(std::is_arithmetic_v<InT>);
    static_assert(filter_size > 1);
    static_assert(std::is_signed_v<SignedInT>);
    static_assert(sizeof(SignedInT) > sizeof(InT));

    // cast to the signed type
    ShiftReg<SignedInT, filter_size> buffer_signed;
    UnrolledLoop<0, filter_size>([&](auto i) {
      buffer_signed[i] = static_cast<SignedInT>(buffer[i]);
    });

    // get the middle index and compute the intensity sigma from it
    constexpr int kMidIdx = filter_size / 2;
    const InT middle_pixel = buffer[kMidIdx];
    const auto sig_i = ComputeIntensitySigma(middle_pixel, params);

    // perform the vertical 1D bilateral filter
    auto output_pixel =
      BilateralFilter1D<SignedInT, filter_size>(buffer_signed, spatial_filter,
                                                params, sig_i);

    // return the result, which is the output pixel, as well as the intensity
    // sigma value (sig_i) and the original pixel which area forwarded to the
    // horizontal kernel for the horizontal 1D bilateral calculation and alpha
    // blending, respectively.
    return OutT(output_pixel, middle_pixel, sig_i);
  }
};

//
// Functor for the row stencil callback.
// This performs the 1D horizontal bilateral filter. It uses the intensity
// sigma (sig_i) that was forwarded from the vertical kernel.
//
template<typename InT, typename OutT, int filter_size,
         int filter_size_eff=(filter_size + 1)/2>
struct HorizontalFunctor {
  auto operator()(int row, int col,
                  ShiftReg<InT, filter_size> buffer,
                  ShiftReg<float, filter_size_eff> spatial_filter,
                  ANRParams params) const {
    // static asserts
    static_assert(std::is_arithmetic_v<OutT>);
    static_assert(filter_size > 1);

    // grab the intensity sigma for the middle pixel (forwarded from the
    // vertical kernel)
    constexpr int kMidIdx = filter_size / 2;
    const float sig_i = buffer[kMidIdx].sig_i;

    // grab just the pixel data, pixel_n is the 'new' pixel forwarded from the
    // vertical kernel
    ShiftReg<float, filter_size> buffer_pixels;
    UnrolledLoop<0, filter_size>([&](auto i) {
      buffer_pixels[i] = buffer[i].pixel_n;
    });

    // perform the horizontal 1D bilateral filter
    auto output_pixel =
      BilateralFilter1D<float, filter_size>(buffer_pixels, spatial_filter,
                                            params, sig_i);

    // alpha blending
    auto original_pixel = buffer[kMidIdx].pixel_o;
    auto output_pixel_alpha = (params.alpha * output_pixel) +
                              (params.one_minus_alpha * original_pixel);

    // return the result casted back to a pixel
    // TODO: round rather than cast???
    return OutT(output_pixel_alpha);
  }
};

//
// Submit all of the ANR kernels (vertical and horizontal)
//
template<typename PixelT, typename IndexT, typename InPipe, 
         typename OutPipe, unsigned filter_size,
         unsigned pixels_per_cycle, unsigned max_cols=4096>
std::vector<event> SubmitANRKernels(queue& q, ANRParams params,
                                    int cols, int rows, int frames) {
  // datatypes
  using VerticalInT = PixelT;
  using VerticalOutT = DataForwardStruct<PixelT>;
  using HorizontalInT = VerticalOutT;  // vertical out is horizontal in
  using HorizontalOutT = PixelT;

  // the internal pipe between the vertical and horizontal kernels
  using IntraPipeT = DataBundle<VerticalOutT, pixels_per_cycle>;
  using IntraPipe = sycl::INTEL::pipe<IntraPipeID, IntraPipeT>;

  // static asserts
  static_assert(filter_size > 1);
  static_assert(max_cols > 1);
  static_assert(pixels_per_cycle > 0);
  static_assert(IsPow2(pixels_per_cycle));
  static_assert(max_cols > pixels_per_cycle);
  static_assert(std::is_arithmetic_v<PixelT>);
  static_assert(std::is_integral_v<IndexT>);

  // validate the arguments
  int padded_cols = PadColumns<IndexT, filter_size>(cols);
  if (cols > max_cols) {
    std::cerr << "ERROR: cols exceeds the maximum (max_cols)"
              << "(" << cols << " > " << max_cols << ")\n";
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
  } else if ((cols % pixels_per_cycle) != 0) {
    std::cerr << "ERROR: the number of columns (" << cols
              << ") must be a multiple of the number of pixels per cycle ("
              << pixels_per_cycle << ")\n";
    std::terminate();
  } else if ((padded_cols % pixels_per_cycle) != 0) {
    std::cerr << "ERROR: the number of padded columns (" << padded_cols
              << ") must be a multiple of the number of pixels per cycle ("
              << pixels_per_cycle << ")\n";
    std::terminate();
  } else if (padded_cols >= std::numeric_limits<IndexT>::max()) {
    // padded_cols >= cols, so just check padded_cols
    std::cerr << "ERROR: the number of padded columns (" << padded_cols
              << ") is too big to be counted to by the IndexType (max="
              << std::numeric_limits<IndexT>::max() << ")\n";
    std::terminate();
  } else if (rows >= std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the number of rows (" << rows
              << ") is too big to be counted to by the IndexType (max="
              << std::numeric_limits<IndexT>::max() << ")\n";
    std::terminate();
  } else if (frames >= std::numeric_limits<IndexT>::max()) {
    std::cerr << "ERROR: the number of frames (" << rows
              << ") is too big to be counted to by the IndexType (max="
              << std::numeric_limits<IndexT>::max() << ")\n";
    std::terminate();
  }

  // cast the rows, columns, and frames to to index type and use these variables
  // inside the kernel to avoid the device dealing with sign conversions
  const IndexT cols_k(cols);
  const IndexT rows_k(rows);
  const IndexT frames_k(frames);
  
  // create the spatial filter for the stencil operation
  constexpr int filter_size_eff = (filter_size + 1) / 2; // ceil(filter_size/2)
  auto spatial_filter = BuildGaussianFilter1D<filter_size_eff>(params.sig_s);

  // the functors for the vertical and horizontal kernels.
  // You can either use a functor or a lamda, both work.
  auto vertical_func =
    VerticalFunctor<VerticalInT, VerticalOutT, filter_size>();
  auto horizontal_func =
    HorizontalFunctor<HorizontalInT, HorizontalOutT, filter_size>();

  // submit the vertical kernel using a column stencil
  auto vertical_kernel = q.submit([&](handler &h) {
    h.single_task<VerticalKernelID>([=] {
      ColumnStencil<VerticalInT, VerticalOutT, IndexT,
                    InPipe, IntraPipe,
                    filter_size, max_cols, pixels_per_cycle>(rows_k, cols_k, frames_k,
                                                             VerticalInT(0),
                                                             vertical_func,
                                                             spatial_filter,
                                                             params);
    });
  });

  // submit the horizontal kernel using a row stencil
  auto horizontal_kernel = q.submit([&](handler &h) {
    h.single_task<HorizontalKernelID>([=] {
      RowStencil<HorizontalInT, HorizontalOutT, IndexT,
                 IntraPipe, OutPipe,
                 filter_size, pixels_per_cycle>(rows_k, cols_k, frames_k,
                                                HorizontalInT(0),
                                                horizontal_func,
                                                spatial_filter, params);
    });
  });
  
  return {vertical_kernel, horizontal_kernel};
}

#endif  /* __ANR_HPP__ */