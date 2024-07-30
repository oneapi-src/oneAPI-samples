#pragma once
#include "bmp_tools.hpp"
#include "rgb_pixels.hpp"

/// @brief Convert RGB pixels as defined by `bmp_tools.hpp` into RGB pixels as
/// defined by `rgb_pixels.hpp`.
/// @param[in] bmp_img A container of pixels as defined by `bmp_tools.hpp`
/// @return A container of pixels as defined in `rgb_pixels.hpp`.
vvp_rgb::ImageRGB ConvertToVvpRgb(const bmp_tools::BitmapRGB &bmp_img) {
  size_t rows = bmp_img.GetRows();
  size_t cols = bmp_img.GetCols();
  size_t pixel_count = rows * cols;
  vvp_rgb::ImageRGB vvp_buf(rows, cols);

  for (size_t idx = 0; idx < pixel_count; idx++) {
    uint32_t pixel_int = bmp_img(idx);
    bmp_tools::PixelRGB bmp_rgb(pixel_int);

    // convert from 8-bit to whatever the VVP IP expects
    vvp_rgb::PixelRGB pixel_vvp{
        (uint16_t)(bmp_rgb.b << (vvp_rgb::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.g << (vvp_rgb::kBitsPerChannel - 8)),   //
        (uint16_t)(bmp_rgb.r << (vvp_rgb::kBitsPerChannel - 8))};  //

    vvp_buf(idx) = pixel_vvp;
  }

  return vvp_buf;
}

/// @brief Convert RGB pixels as defined by `rgb_pixels.hpp` into RGB pixels as
/// defined by `bmp_tools.hpp`.
/// @param[in] vvp_buf A container of pixels as defined in `rgb_pixels.hpp`.
/// @return A container of pixels as defined in `bmp_tools.hpp`.
bmp_tools::BitmapRGB ConvertToBmpRgb(const vvp_rgb::ImageRGB &vvp_buf) {
  size_t rows = vvp_buf.GetRows();
  size_t cols = vvp_buf.GetCols();
  size_t pixel_count = rows * cols;
  bmp_tools::BitmapRGB bmp_img(rows, cols);

  for (size_t idx = 0; idx < pixel_count; idx++) {
    vvp_rgb::PixelRGB pixel_conv = vvp_buf(idx);

    // convert the VVP IP back to 8-bit
    bmp_tools::PixelRGB bmp_rgb(
        (uint8_t)(pixel_conv.r >> (vvp_rgb::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.g >> (vvp_rgb::kBitsPerChannel - 8)),   //
        (uint8_t)(pixel_conv.b >> (vvp_rgb::kBitsPerChannel - 8)));  //

    uint32_t pixel_int = bmp_rgb.GetImgPixel();
    bmp_img(idx) = pixel_int;
  }

  return bmp_img;
}