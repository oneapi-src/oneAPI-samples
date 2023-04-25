//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>

#include "utils.hpp"

using namespace sycl;
using namespace std;

int main() {
  // Image size is width x height
  int width = 1440;
  int height = 960;

  Img<ImgFormat::BMP> image{width, height};
  ImgFractal fractal{width, height};

  // Lambda to process image with gamma = 2
  auto gamma_f = [](ImgPixel &pixel) {
    auto v = (0.3f * pixel.r + 0.59f * pixel.g + 0.11f * pixel.b) / 255.0f;

    auto gamma_pixel = static_cast<uint8_t>(255 * v * v);
    if (gamma_pixel > 255) gamma_pixel = 255;
    pixel.set(gamma_pixel, gamma_pixel, gamma_pixel, gamma_pixel);
  };

  // fill image with created fractal
  int index = 0;
  image.fill([&index, width, &fractal](ImgPixel &pixel) {
    int x = index % width;
    int y = index / width;

    auto fractal_pixel = fractal(x, y);
    if (fractal_pixel < 0) fractal_pixel = 0;
    if (fractal_pixel > 255) fractal_pixel = 255;
    pixel.set(fractal_pixel, fractal_pixel, fractal_pixel, fractal_pixel);

    ++index;
  });

  string original_image = "fractal_original.png";
  string processed_image = "fractal_gamma.png";
  Img<ImgFormat::BMP> image2 = image;
  image.write(original_image);

  // call standard serial function for correctness check
  image.fill(gamma_f);

  // use default policy for algorithms execution
  auto policy = oneapi::dpl::execution::dpcpp_default;
  // We need to have the scope to have data in image2 after buffer's destruction
  {
    //# STEP 1: Create a buffer for the image2
    buffer<ImgPixel> b(image2.data(), image2.width() * image2.height());

    //# STEP 2: Create buffer iterators for the buffer that was created
    auto b_begin = oneapi::dpl::begin(b);
    auto b_end = oneapi::dpl::end(b);

    //# STEP 3: Call std::for_each with buffer iterators and call the gamma correction algorithm 
    std::for_each(policy, b_begin, b_end, gamma_f);
  }

  image2.write(processed_image);
  // check correctness
  if (check(image.begin(), image.end(), image2.begin())) {
    cout << "success\n";
  } else {
    cout << "fail\n";
    return 1;
  }
  cout << "Run on "
       << policy.queue().get_device().template get_info<info::device::name>()
       << "\n";
  cout << "Original image is in " << original_image << "\n";
  cout << "Image after applying gamma correction on the device is in "
       << processed_image << "\n";

  return 0;
}
