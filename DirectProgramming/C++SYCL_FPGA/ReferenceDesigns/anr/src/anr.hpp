#ifndef __ANR_HPP__
#define __ANR_HPP__

//
// This file contains a bulk of the functionality for the ANR design on the
// on the device. It contains the logic to submit the various kernels for the
// ANR pipeline.
//

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "anr_params.hpp"
#include "column_stencil.hpp"
#include "constants.hpp"
#include "data_bundle.hpp"
#include "intensity_sigma_lut.hpp"
#include "qfp.hpp"
#include "qfp_exp_lut.hpp"
#include "qfp_inv_lut.hpp"
#include "row_stencil.hpp"
#include "shift_reg.hpp"

// Included from DirectProgramming/C++SYCL_FPGA/include/
#include "constexpr_math.hpp"
#include "unrolled_loop.hpp"

using namespace sycl;

// declare the kernel and pipe names globally to reduce name mangling
class IntraPipeID;
class VerticalKernelID;
class HorizontalKernelID;

//
// A struct to carry the new (i.e., current) pixel, the original pixel, and the
// intensity sigma from the vertical to horizontal kernel
//
struct DataForwardStruct {
  DataForwardStruct() {}
  DataForwardStruct(PixelT pixel_n) : pixel_n(pixel_n) {}
  DataForwardStruct(PixelT pixel_n, PixelT pixel_o, float sig_i)
      : pixel_n(pixel_n), pixel_o(pixel_o), sig_i(sig_i) {}

  PixelT pixel_n;   // the new pixel
  PixelT pixel_o;   // the original pixel
  float sig_i;      // the intensity sigma value for the pixel
};

//
// Build the power values for a 1D gaussian filter. The 'actual' gaussian values
// are exp(-x) where 'x' are the 'powers' in the 'filter'.
//
template <int size>
auto BuildGaussianPowers1D(float sigma) {
  fpga_tools::ShiftReg<float, size> filter;
  for (int x = -size / 2; x <= size / 2; x++) {
    float x_over_sig = x / sigma;
    filter[x + size / 2] = 0.5 * x_over_sig * x_over_sig; // 0.5*(x/sigma)^2
  }
  return filter;
}

//
// Given a float value, convert it to an unsigned pixel value by saturating it
// in the extremes (i.e., min/max).
//
PixelT Saturate(float pixel_float) {
  constexpr unsigned kMaxPixelVal = std::numeric_limits<PixelT>::max();
  constexpr unsigned kMinPixelVal = std::numeric_limits<PixelT>::min();
  constexpr unsigned kFloatSignOffset = ((sizeof(float) * 8) - 1);

  // get the bits of the float for the negative check
  unsigned int pixel_float_bits = reinterpret_cast<unsigned int&>(pixel_float);

  PixelT pixel;
  if (pixel_float >= kMaxPixelVal) {
    pixel = kMaxPixelVal;
  } else if ((pixel_float_bits >> kFloatSignOffset) & 0x1) { // pixel_float < 0
    pixel = kMinPixelVal;
  } else {
    pixel = pixel_float;
  }
  return pixel;
}

//
// Computes the 1D bilateral filter. The spatial filter is passed as an
// argument and the intensity filter is computed based on the pixel window
// ('buffer') and the ANR parameters ('params'). Together, the spatial and
// intensity filters create the bilateral filter.
//
template <int filter_size, int filter_size_eff = (filter_size + 1) / 2>
inline float
BilateralFilter1D(fpga_tools::ShiftReg<PixelT, filter_size>& buffer,
                  fpga_tools::ShiftReg<float, filter_size_eff>& spatial_power,
                  ANRParams params, float sig_i_inv_squared_x_half,
                  const ExpLUT& exp_lut,
                  const InvLUT& inv_lut) {
  // We need to hold all pixels bits, but need the type to be signed for the
  // BilateralFilter1D since it subtracts the values. So add one bit to it and
  // make it a signed ac_int.
  using SignedPixelT = ac_int<kPixelBits + 1, true>;

  // the middle pixel index
  constexpr int mid_idx = filter_size / 2;

  // static asserts
  static_assert(filter_size > 1);

  // convert unsigned pixels to signed
  fpga_tools::ShiftReg<SignedPixelT, filter_size> buffer_signed;
  fpga_tools::UnrolledLoop<filter_size>([&](auto i) {
    buffer_signed[i] = static_cast<SignedPixelT>(buffer[i]);
  });

  // build the bilateral filter
  fpga_tools::ShiftReg<float, filter_size_eff> bilateral_filter;
  float filter_sum = 0.0;

  fpga_tools::UnrolledLoop<filter_size_eff>([&](auto i) {
    // get the absolute value of the pixel differences
    float intensity_diff_squared;
    if constexpr (mid_idx == (i * 2)) {
      // special case for middle pixel, the absolute difference will be 0
      // and therefore the absolute value difference squared will also be 0
      intensity_diff_squared = 0.0f;
    } else {
      // compute differences squared
      const SignedPixelT intensity_diff =
          buffer_signed[mid_idx] - buffer_signed[i * 2];
      intensity_diff_squared = intensity_diff * intensity_diff;
    }

    // compute the filter value as e^-(intensity_component + spatial_component)
    // Use a LUT to compute the exp(-x) value
    float filter_val;
    if constexpr (mid_idx == (i * 2)) {
      // (buffer[mid_idx] - buffer[i * 2]) = 0 in this case, so the intensity
      // component is 0. For similar reasons, the spatial component is also 0
      // and therefore filter_val is e^(-0) = 1.
      filter_val = 1.0f;
    } else {
      // intensity_component = 1/2 * (intensity_diff/sig_i)^2
      // = 1/2*(1/sig_i)^2 *(intensity_diff)^2
      // We have precomputed 1/2*(1/sig_i)^2 in the host
      const float intensity_component =
          intensity_diff_squared * sig_i_inv_squared_x_half;

      // spatial component is a regular Gaussian that was precomputed using
      // the 'BuildGaussianPowers1D' function
      const float spatial_component = spatial_power[i];

      // the bilateral filter power value, where the actual bilateral filter
      // value is e^-(exp_power)
      const float exp_power = intensity_component + spatial_component;

      // now that we have the exponential power value ('exp_power'), use the
      // exponential LUT ('ExpLUT') to lookup the result of exp(-exp_power).
      // NOTE: when creating the exponential LUT, we stored the values of
      // exp(-x) = 1/exp(x). This avoids negating the value of 'exp_power'.
      const auto exp_lut_idx = ExpLUT::QFP::FromFP32(exp_power);
      filter_val = ExpLUT::QFP::ToFP32(exp_lut[exp_lut_idx]);
    }

    // compute the bilateral filter value
    bilateral_filter[i] = filter_val;
    filter_sum += filter_val;
  });

  // Convolve the 1D bilateral filter with the pixel window
  float filtered_pixel = 0.0;
  fpga_tools::UnrolledLoop<filter_size_eff>([&](auto i) {
    filtered_pixel += (float(buffer[i * 2]) * bilateral_filter[i]);
  });
  
  // Normalize the pixel value by the bilateral filter sum. Use the inverse
  // LUT to compute 1/filter_sum. This saves area by using a 32-bit
  // floating-point multiplication, instead of division.
  // Computes: filtered_pixel /= filter_sum
  const auto inv_lut_idx = InvLUT::QFP::FromFP32(filter_sum);
  filtered_pixel *= InvLUT::QFP::ToFP32(inv_lut[inv_lut_idx]);

  return filtered_pixel;
}

//
// Functor for the column stencil callback.
// This performs the 1D vertical bilateral filter. It also computes the
// intensity sigma value (sig_i) and bundles it with the original and partially
// filtered pixel to be forwarded to the horizontal kernel.
//
template <int filter_size, int filter_size_eff = (filter_size + 1) / 2>
struct VerticalFunctor {
  auto operator()(int row, int col,
                 fpga_tools::ShiftReg<PixelT, filter_size> buffer,
                 fpga_tools::ShiftReg<float, filter_size_eff> spatial_power,
                 ANRParams params, const ExpLUT& exp_lut,
                 const InvLUT& inv_lut, IntensitySigmaLUT& sig_i_lut) const {
    // static asserts to validate template arguments
    static_assert(filter_size > 1);

    // get the middle index and compute the intensity sigma from it
    constexpr int mid_idx = filter_size / 2;
    const PixelT middle_pixel = buffer[mid_idx];
    const auto sig_i_inv_squared_x_half = sig_i_lut[middle_pixel];

    // perform the vertical 1D bilateral filter
    auto output_pixel_float = BilateralFilter1D<filter_size>(
        buffer, spatial_power, params, sig_i_inv_squared_x_half,
        exp_lut, inv_lut);

    // saturate the output pixel
    PixelT output_pixel = Saturate(output_pixel_float);

    // return the result, which is the output pixel, as well as the intensity
    // sigma value (sig_i) and the original pixel, which are forwarded to the
    // horizontal kernel for the horizontal 1D bilateral calculation and alpha
    // blending, respectively.
    return DataForwardStruct(output_pixel, middle_pixel,
                             sig_i_inv_squared_x_half);
  }
};

//
// Functor for the row stencil callback.
// This performs the 1D horizontal bilateral filter. It uses the intensity
// sigma (sig_i) that was forwarded from the vertical kernel.
//
template <int filter_size, int filter_size_eff = (filter_size + 1) / 2>
struct HorizontalFunctor {
  auto operator()(int row, int col,
                  fpga_tools::ShiftReg<DataForwardStruct, filter_size> buffer,
                  fpga_tools::ShiftReg<float, filter_size_eff> spatial_power,
                  ANRParams params, ANRParams::AlphaFixedT alpha_fixed,
                  ANRParams::AlphaFixedT one_minus_alpha_fixed,
                  const ExpLUT& exp_lut, const InvLUT& inv_lut) const {    
    // static asserts
    static_assert(filter_size > 1);

    // grab the intensity sigma for the middle pixel (forwarded from the
    // vertical kernel)
    constexpr int mid_idx = filter_size / 2;
    const float sig_i_inv_squared_x_half = buffer[mid_idx].sig_i;

    // grab just the pixel data, pixel_n is the 'new' pixel forwarded from the
    // vertical kernel (i.e., the partially filtered one)
    fpga_tools::ShiftReg<PixelT, filter_size> buffer_pixels;
    fpga_tools::UnrolledLoop<filter_size>([&](auto i) {
      buffer_pixels[i] = buffer[i].pixel_n;
    });

    // perform the horizontal 1D bilateral filter
    auto output_pixel_float = BilateralFilter1D<filter_size>(
        buffer_pixels, spatial_power, params, sig_i_inv_squared_x_half,
        exp_lut, inv_lut);

    // saturate the output pixel
    PixelT output_pixel = Saturate(output_pixel_float);

    // fixed-point alpha blending with the original pixel
    const PixelT original_pixel(buffer[mid_idx].pixel_o);
    auto output_pixel_alpha =
        (alpha_fixed * output_pixel) + (one_minus_alpha_fixed * original_pixel);
    auto output_pixel_tmp = output_pixel_alpha.to_ac_int();

    // return the result casted back to a pixel
    return PixelT(output_pixel_tmp);
  }
};

//
// Submit all of the ANR kernels (vertical and horizontal)
//
template <typename IndexT, typename InPipe, typename OutPipe,
          unsigned filter_size, unsigned pixels_per_cycle,
          unsigned max_cols>
std::vector<event> SubmitANRKernels(queue& q, int cols, int rows,
                                    ANRParams params,
                                    float* sig_i_lut_data_ptr) {
  // the internal pipe between the vertical and horizontal kernels
  using IntraPipeT =
      fpga_tools::DataBundle<DataForwardStruct, pixels_per_cycle>;
  using IntraPipe = ext::intel::pipe<IntraPipeID, IntraPipeT>;

  // static asserts to validate template arguments
  static_assert(filter_size > 1);
  static_assert(max_cols > 1);
  static_assert(pixels_per_cycle > 0);
  static_assert(fpga_tools::IsPow2(pixels_per_cycle));
  static_assert(max_cols > pixels_per_cycle);
  static_assert(std::is_integral_v<IndexT>);

  // validate the function arguments
  int padded_cols = PadColumns<IndexT, filter_size>(cols);
  if (cols > max_cols) {
    std::cerr << "ERROR: cols exceeds the maximum (max_cols) "
              << "(" << cols << " > " << max_cols << ")\n";
    std::terminate();
  } else if (cols <= 0) {
    std::cerr << "ERROR: cols must be strictly positive\n";
    std::terminate();
  } else if (rows <= 0) {
    std::cerr << "ERROR: rows must be strictly positive\n";
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
  }

  // cast the rows and columns to the index type and use these
  // variables inside the kernel to avoid the device dealing with conversions
  const IndexT cols_k(cols);
  const IndexT rows_k(rows);

  // create the spatial filter for the stencil operation
  constexpr int filter_size_eff = (filter_size + 1) / 2;  // ceil(filter_size/2)
  auto spatial_power = BuildGaussianPowers1D<filter_size_eff>(params.sig_s);

  // Functors or lambdas can be used for the vertical and horizontal kernels.
  auto vertical_func = VerticalFunctor<filter_size>();
  auto horizontal_func = HorizontalFunctor<filter_size>();

  // submit the vertical kernel using a column stencil
  auto vertical_kernel = q.single_task<VerticalKernelID>([=] {
    // copy host side intensity sigma LUT to the device
    IntensitySigmaLUT sig_i_lut(sig_i_lut_data_ptr);

    // build the constexpr exp() and inverse LUT ROMs
    constexpr ExpLUT exp_lut;
    constexpr InvLUT inv_lut;

    // Start the column stencil.
    // It will callback to 'vertical_func' with all of the additional
    // arguments listed after 'vertical_func' (i.e., spatial_power,
    // params, ...)
    ColumnStencil<PixelT, DataForwardStruct, IndexT, InPipe,
                  IntraPipe, filter_size, max_cols, pixels_per_cycle>(rows_k,
                  cols_k, PixelT(0), vertical_func, spatial_power, params,
                  std::cref(exp_lut), std::cref(inv_lut),
                  std::ref(sig_i_lut));
  });

  // submit the horizontal kernel using a row stencil
  auto horizontal_kernel = q.single_task<HorizontalKernelID>([=] {
    // build the constexpr exp() and inverse LUT ROMs
    constexpr ExpLUT exp_lut;
    constexpr InvLUT inv_lut;

#ifdef IP_MODE
    ANRParams::AlphaFixedT alpha_fixed(0.75);
    ANRParams::AlphaFixedT one_minus_alpha_fixed(0.25);
#else
    // convert the alpha and (1-alpha) values to fixed-point
    ANRParams::AlphaFixedT alpha_fixed(params.alpha);
    ANRParams::AlphaFixedT one_minus_alpha_fixed(params.one_minus_alpha);
#endif
    
    // Start the row stencil.
    // It will callback to 'horizontal_func' with the additional all of the
    // additional arguments listed after 'horizontal_func' (i.e.,
    // spatial_power, params, alpha_fixed, ...)
    RowStencil<DataForwardStruct, PixelT, IndexT, IntraPipe, OutPipe,
                filter_size, pixels_per_cycle>(rows_k, cols_k,
                DataForwardStruct(0), horizontal_func, spatial_power,
                params, alpha_fixed, one_minus_alpha_fixed, std::cref(exp_lut),
                std::cref(inv_lut));
  });

  return {vertical_kernel, horizontal_kernel};
}

#endif /* __ANR_HPP__ */
