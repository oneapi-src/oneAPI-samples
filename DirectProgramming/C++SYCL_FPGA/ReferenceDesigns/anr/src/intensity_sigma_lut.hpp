#ifndef __INTENSITY_SIGMA_LUT_HPP__
#define __INTENSITY_SIGMA_LUT_HPP__

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <type_traits>

#include "anr_params.hpp"
#include "constants.hpp"

//
// A LUT for computing the intensity sigma value of a pixel
//
class IntensitySigmaLUT {
 public:
  // default constructor
  IntensitySigmaLUT() {}

#if defined (IS_BSP)
  // construct from a device_ptr (for constructing from device memory)
  IntensitySigmaLUT(device_ptr<float> ptr) {
    // use a pipelined LSU to load from device memory since we don't
    // care about the performance of the copy.
    using PipelinedLSU = ext::intel::lsu<>;
    for (int i = 0; i < lut_depth; i++) {
      data_[i] = PipelinedLSU::load(ptr + i);
    }
  }
#else 
  // construct from a regular pointer
  IntensitySigmaLUT(float* ptr) {
    for (int i = 0; i < lut_depth; i++) {
      data_[i] = ptr[i];
    }
  }
#endif

  // construct from the ANR parameters (actually builds the LUT)
  IntensitySigmaLUT(ANRParams params) {
    for (int i = 0; i < lut_depth; i++) {
      float sig_i = sycl::sqrt(params.k * float(i) + params.sig_shot_2) *
                    params.sig_i_coeff;
      float sig_i_inv = 1.0f / sig_i;
      float sig_i_inv_squared = sig_i_inv * sig_i_inv;
      float sig_i_inv_squared_2 = 0.5f * sig_i_inv_squared;
      data_[i] = sig_i_inv_squared_2;  // storing 0.5 * (1/sig_i)^2
    }
  }

  // helper static method to allocate enough memory to hold the LUT
  static float* Allocate(sycl::queue& q) {
#if defined (IS_BSP)
    float* ptr = sycl::malloc_device<float>(lut_depth, q);
#else 
    float* ptr = sycl::malloc_shared<float>(lut_depth, q);
#endif   
    if (ptr == nullptr) {
      std::cerr << "ERROR: could not allocate space for 'ptr'\n";
      std::terminate();
    }
    return ptr;
  }

  // helper method to copy the data to the device
  sycl::event CopyData(sycl::queue& q, float* ptr) {
    return q.memcpy(ptr, data_, lut_depth * sizeof(float));
  }

  const float& operator[](int i) const { return data_[i]; }

 private:
  static constexpr int lut_depth = std::numeric_limits<PixelT>::max() -
                                   std::numeric_limits<PixelT>::min() + 1;
  float data_[lut_depth];
};

#endif /* __INTENSITY_SIGMA_LUT_HPP__ */