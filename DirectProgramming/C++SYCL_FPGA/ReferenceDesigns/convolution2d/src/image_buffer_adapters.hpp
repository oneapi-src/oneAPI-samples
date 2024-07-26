#pragma once
#include "bmp_tools.hpp"
#include "convolution_types.hpp"

/// @brief Initialize a buffer meant to store an image. This helps with
/// debugging, because you can see if an image was only partly written. The
/// buffer is initialized with an incrementing pattern, ranging from `0` to
/// `(1 << 24)`.
/// @param[out] buf Buffer to initialize
/// @param[in] size Number of pixels to initialize in the buffer
void InitializeBuffer(conv2d::PixelRGB *buf, size_t size) {
  uint16_t pixel = 0;
  for (size_t i = 0; i < size; i++) {
    pixel++;
    buf[i] = conv2d::PixelRGB{pixel, pixel, pixel};
  }
}

/// @brief Convert pixels read from a bmp image using the bmp_tools functions to
/// pixels that can be parsed by our 2D convolution IP.
/// @param[in] bmp_img pixels read by bmptools
/// @param[out] vvp_buf pixels to be consumed by 2D convolution IP
/// @param[in] pixel_count (input) number of pixels in input image and output
/// image
void ConvertToVvpRgb(bmp_tools::BitmapRGB bmp_img, conv2d::PixelRGB *vvp_buf,
                     size_t pixel_count) {
  std::cout << "INFO: convert to vvp type." << std::endl;
  for (size_t idx = 0; idx < pixel_count; idx++) {
    uint32_t pixel_int = bmp_img(idx);
    bmp_tools::PixelRGB bmp_rgb(pixel_int);

    // convert from 8-bit to whatever the VVP IP expects
    conv2d::PixelRGB pixel_vvp{
        (uint16_t)(bmp_rgb.b << (conv2d::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.g << (conv2d::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.r << (conv2d::kBitsPerChannel - 8))};  //

    vvp_buf[idx] = pixel_vvp;
  }
}

/// @brief Convert pixels read from the 2D convolution IP to a format that can
/// be read by the bmptools functions.
/// @param[in] vvp_buf Pixels produced by 2D convolution IP
/// @param[in] rows Number of rows in input image and output image
/// @param[in] cols Number of columns in input image and output image
/// @return bmp_buf Pixels to send to bmptools
bmp_tools::BitmapRGB ConvertToBmpRgb(conv2d::PixelRGB *vvp_buf, size_t rows,
                                     size_t cols) {
  std::cout << "INFO: convert to bmp type." << std::endl;
  size_t pixel_count = rows * cols;
  bmp_tools::BitmapRGB bmp_img(rows, cols);
  for (size_t idx = 0; idx < pixel_count; idx++) {
    conv2d::PixelRGB pixel_conv = vvp_buf[idx];

    // convert the VVP IP back to 8-bit
    bmp_tools::PixelRGB bmp_rgb(
        (uint8_t)(pixel_conv.r >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.g >> (conv2d::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.b >> (conv2d::kBitsPerChannel - 8)));  //

    uint32_t pixel_int = bmp_rgb.GetImgPixel();
    bmp_img(idx) = pixel_int;
  }

  return bmp_img;
}