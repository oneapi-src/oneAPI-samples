#ifndef __INTENSITY_SIGMA_LUT_HPP__
#define __INTENSITY_SIGMA_LUT_HPP__

#include <type_traits>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#include "anr_params.hpp"

//
// A LUT for computing the sigma value of a pixel
//
template<typename PixelT>
class IntensitySigmaLUT {
  static_assert(std::is_unsigned_v<PixelT>);
public:
  IntensitySigmaLUT() {}

  IntensitySigmaLUT(device_ptr<float> ptr) {
    // use a pipelined LSU to load from device memory since we don't
    // care about the performance of the copy.
    using PipelinedLSU = INTEL::lsu<>;
    for (int i = 0; i < lut_depth; i++) {
      data_[i] = PipelinedLSU::load(ptr + i);
    }
  }

  IntensitySigmaLUT(ANRParams params) {
    for (int i = 0; i < lut_depth; i++) {
      float sig_i =
        sycl::sqrt(params.k * float(i) + params.sig_shot_2) * params.sig_i_coeff;
      float sig_i_inv = 1.0f / sig_i;
      float sig_i_inv_squared = sig_i_inv * sig_i_inv;
      float sig_i_inv_squared_2 = 0.5f * sig_i_inv_squared;
      data_[i] = sig_i_inv_squared_2;  // storing 0.5 * (1/sig_i)^2
    }
  }

  static float* AllocateDevice(sycl::queue& q) {
    float *ptr = sycl::malloc_device<float>(lut_depth, q);
    if (ptr == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'ptr'\n";
      std::terminate();
    }
    return ptr;
  }

  sycl::event CopyDataToDevice(sycl::queue& q, float* ptr) {
    return q.memcpy(ptr, data_, lut_depth * sizeof(float));
  }

  const float& operator[](int i) const { return data_[i]; }

private:
  static constexpr int lut_depth = std::numeric_limits<PixelT>::max() -
                                  std::numeric_limits<PixelT>::min() + 1;
  float data_[lut_depth];
};

#endif /* __INTENSITY_SIGMA_LUT_HPP__ */