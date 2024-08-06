//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// bmp_tools.h

// This header file parses bitmap image files. It uses C++ standard libraries,
// and should only be used in host code.

#ifndef BMP_TOOLS_H
#define BMP_TOOLS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "matrix2d_host.hpp"

#ifndef FILENAME_BUF_SIZE
#if defined(_WIN32) || defined(_WIN64)
#define FILENAME_BUF_SIZE _MAX_PATH
#else
#define FILENAME_BUF_SIZE MAX_PATH
#endif
#endif

namespace bmp_tools {

// file extension to use when reading/writing bitmap files
constexpr char kFileExtension[] = "bmp";

/// @brief Store an image that can be read from or written to a .bmp file. The
/// individual pixel values may be changed at runtime, but the dimensions are
/// fixed.
using BitmapRGB = Matrix2d<unsigned int>;

/// @brief This convenience struct lets you manipulate color channels within
/// pixels used by bmp_tools functions.
struct PixelRGB {
  // blue is least significant
  uint8_t b;
  uint8_t g;
  uint8_t r;
  uint8_t reserved;

  /// @brief Construct a `bmp_tools::PixelRGB` using an unsigned 32-bit pixel
  /// used by `bmp_tools` functions.
  /// @param img_pixel An unsigned 32-bit pixel used by `bmp_tools` functions
  PixelRGB(uint32_t img_pixel) {
    b = (img_pixel >> 0) & 0xff;
    g = (img_pixel >> 8) & 0xff;
    r = (img_pixel >> 16) & 0xff;
  }

  /// @brief Default constructor that initializes all color channels to `0`.
  PixelRGB() { PixelRGB(0); }

  /// @brief Construct a `bmp_tools::PixelRGB` using members
  /// @param m_r Red color channel
  /// @param m_g Green color channel
  /// @param m_b Blue color channel
  PixelRGB(uint8_t m_r, uint8_t m_g, uint8_t m_b)
      : b(m_b), g(m_g), r(m_r), reserved(0) {}

  /// @brief Transform a `bmp_tools::PixelRGB` into an unsigned 32-bit pixel
  /// used by `bmp_tools` functions.
  /// @return An unsigned 32-bit pixel used by `bmp_tools` functions.
  uint32_t GetImgPixel() {
    uint32_t img_pixel =
        ((b & 0xff) << 0) + ((g & 0xff) << 8) + ((r & 0xff) << 16);
    return img_pixel;
  }

  /// @brief Check that two pixels are similar to one another, that is, that
  /// they differ by no more than `thresh`.
  /// @param other The pixel to compare against
  /// @param threshold Maximum amount by any two color channels may deviate
  /// @return `true` if `this` and `other` differ by no more than `thresh`
  bool CheckSimilarity(PixelRGB other, unsigned char threshold) {
    bool similar = true;
    if (abs(r - other.r) > threshold) similar = false;
    if (abs(g - other.g) > threshold) similar = false;
    if (abs(b - other.b) > threshold) similar = false;

    return similar;
  }
};

enum BmpHeaderField : uint16_t {
  BM = 0x4D42,  // "BM"
  BA = 0x4142,  // "BA"
  CI = 0x4943,  // "CI"
  CP = 0x5043,  // "CP"
  IC = 0x4349,  // "IC"
  PT = 0x5450   // "PT"
};

// pack this struct at the byte level so we can load file data directly into it
#pragma pack(push, 1)
struct BmpFileHeader {
  BmpHeaderField header_field;
  uint32_t file_size;
  uint16_t reserved_0;
  uint16_t reserved_1;
  uint32_t img_data_offset;
};
#pragma pack(pop)

// pack this struct at the byte level so we can load file data directly into it
#pragma pack(push, 1)
struct WindowsBitmapInfoHeader {
  uint32_t header_size;  // should be 40 bytes
  int32_t img_width;
  int32_t img_height;
  uint16_t img_planes;  // must be 1
  uint16_t img_bit_depth;
  uint32_t img_compression;
  uint32_t img_size;
  uint32_t img_resolution_horiz;
  uint32_t img_resolution_vert;
  uint32_t img_colors;
  uint32_t img_important_colors;
};
#pragma pack(pop)

// Bitmap header size in bytes
static constexpr int BMP_HEADER_SIZE =
    sizeof(WindowsBitmapInfoHeader) + sizeof(BmpFileHeader);

enum BmpError {
  OK = 0,
  INVALID_FILE = 1 << 0,
  UNSUPPORTED_HEADER = 1 << 1,
  UNSUPPORTED_BIT_DEPTH = 1 << 2,
  UNSUPPORTED_PLANES = 1 << 3,
  UNSUPPORTED_NUM_COLORS = 1 << 4,
  UNSUPPORTED_IMPORTANT_COLORS = 1 << 5,
  INVALID_DIMENSIONS = 1 << 6,
  FILE_IO_ERROR = 1 << 7
};

/// @brief Read a `.bmp` file pointed to by `filename` into memory. Each image
/// pixel will be stored as a 32-bit unsigned integer.
///
/// @paragraph For simplicity, we only support a certain type of BMP file,
/// namely 24-bit Windows-style, with all important colors and a single color
/// plane.
///
/// @param[in] file_path File path to read from
///
/// @param[out] error_code Error code. If everything is OK, this will match
/// BmpError::OK. You can do bitwise comparisons with the members of the
/// enum `BmpError` to determine which error(s) occurred.
///
/// @return A `BitmapRGB` containing the pixels of the image read in from
/// `file_path`. In case an error occurs, return a bitmap of size 0.
inline BitmapRGB ReadBmp(std::string &file_path, unsigned int &error_code) {
  BitmapRGB bitmap_error(0, 0);
  error_code = BmpError::OK;

  std::ifstream input_bmp;
  input_bmp.open(file_path, std::ios::in | std::ios::binary);
  if (!input_bmp) {
    error_code |= BmpError::INVALID_FILE;
    std::cerr << "ERROR: input file " << file_path << " does not exist."
              << std::endl;
    return bitmap_error;
  }

  // load file header
  BmpFileHeader file_header;
  input_bmp.read(reinterpret_cast<char *>(&file_header), sizeof(BmpFileHeader));

  if (file_header.header_field != BmpHeaderField::BM) {
    error_code |= BmpError::UNSUPPORTED_HEADER;
    std::cerr << "ERROR: only Windows-format bitmap header is supported. "
                 "Please convert to a Windows-style bitmap."
              << std::endl;
    return bitmap_error;
  }

  WindowsBitmapInfoHeader dib_header;
  input_bmp.read(reinterpret_cast<char *>(&dib_header),
                 sizeof(WindowsBitmapInfoHeader));

  int32_t width = dib_header.img_width;
  int32_t height = dib_header.img_height;

  // sanity check that inputs are valid
  if (dib_header.img_bit_depth != 24) {
    error_code |= BmpError::UNSUPPORTED_BIT_DEPTH;
    std::cerr << "ERROR: Only 24-bit BMP is supported. Please ensure your BMP "
                 "uses 24-bit pixels (24)"
              << std::endl;
    return bitmap_error;
  }

  if (dib_header.img_planes != 1) {
    error_code |= BmpError::UNSUPPORTED_PLANES;
    std::cerr << "ERROR: Only 1-plane BMP is supported. Please ensure your BMP "
                 "uses a single color plane (1)."
              << std::endl;
    return bitmap_error;
  }

  if (dib_header.img_colors != 0) {
    error_code |= BmpError::UNSUPPORTED_NUM_COLORS;
    std::cerr << "ERROR: requires 2^n colors. Please ensure your BMP uses the "
                 "default number of colors (0)."
              << std::endl;
    return bitmap_error;
  }

  if (dib_header.img_important_colors != 0) {
    error_code |= BmpError::UNSUPPORTED_IMPORTANT_COLORS;
    std::cerr
        << "ERROR: all colors should be important. Please ensure your BMP uses "
           "the default number of important colors (0)."
        << std::endl;
    return bitmap_error;
  }

  int32_t total_pixels = width * height;
  bool did_overflow = (total_pixels < width) || (total_pixels < height);
  // check that width*height is also valid
  if ((height < 0) || (width < 0) || (total_pixels < 0) || did_overflow) {
    error_code |= BmpError::INVALID_DIMENSIONS;
    std::cerr << "ERROR: got height " << height << ", width " << width
              << std::endl;
    return bitmap_error;
  }

  BitmapRGB bitmap_data(height, width);

  // scroll to image data
  auto read_bytes = sizeof(WindowsBitmapInfoHeader) + sizeof(BmpFileHeader);
  while (read_bytes < file_header.img_data_offset) {
    input_bmp.get();
    read_bytes++;
  }

  // BMP: Each line must be a multiple of 4 bytes
  int padding = (4 - ((width * 3) & 3)) & 3;

  bool failed = false;

  // Color order is BGR, read across bottom row, then repeat going up rows
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      unsigned char b = input_bmp.get();  // B
      unsigned char g = input_bmp.get();  // G

      std::ios_base::iostate state = input_bmp.rdstate();

      bool earlyEOF = state & std::ifstream::eofbit;
      unsigned char r = input_bmp.get();  // R
      bitmap_data(i, j) = PixelRGB(r, g, b).GetImgPixel();
      if (earlyEOF) failed |= earlyEOF;

      state = input_bmp.rdstate();
      bool failBit = state & std::ifstream::failbit;

      if (failBit) failed |= failBit;
    }
    // Discard the padding bytes
    for (int j = 0; j < padding; j++) {
      input_bmp.get();
      bool earlyEOF = input_bmp.rdstate() & std::ifstream::eofbit;
      if (earlyEOF) failed |= earlyEOF;

      bool failBit = input_bmp.rdstate() & std::ifstream::failbit;
      if (failBit) failed |= failBit;
    }
  }
  input_bmp.close();

  if (failed) {
    error_code |= BmpError::FILE_IO_ERROR;
    std::cerr << "ERROR: File I/O error" << std::endl;
    return bitmap_error;
  }

  return bitmap_data;
}

/// @brief Store pixels in `img_data` array to a bitmap pointed to by `filename`
/// @paragraph For simplicity, we only support a certain type of BMP file,
/// namely 24-bit Windows-style, with all important colors and a single color
/// plane.
/// @param[in] file_path Filepath to write to.
/// @param[in] bitmap_rgb Data to write out to file.
/// @param[out] error_code Error code. If everything is OK, this will match
/// BmpError::OK. You can do bitwise comparisons with the members of the
/// enum `BmpError` to determine which error(s) occurred.
inline void WriteBmp(std::string &file_path, BitmapRGB bitmap_rgb,
                     unsigned int &error_code) {
  error_code = BmpError::OK;

  size_t width = bitmap_rgb.GetCols();
  size_t height = bitmap_rgb.GetRows();
  // sanity check that inputs are valid, check that width*height is also valid
  if ((height < 0 || width < 0) || (height * width > (1 << 30))) {
    error_code |= BmpError::INVALID_DIMENSIONS;
    std::cerr << "ERROR: height " << height << ", width " << width << std::endl;
    return;
  }

  unsigned int file_size = width * height * 3 + BMP_HEADER_SIZE;
  unsigned char header[BMP_HEADER_SIZE] = {
      0x42, 0x4d,  // BMP & DIB

      // size of file in bytes
      (static_cast<unsigned char>((file_size >> 0) & 0xff)),
      (static_cast<unsigned char>((file_size >> 8) & 0xff)),
      (static_cast<unsigned char>((file_size >> 16) & 0xff)),
      (static_cast<unsigned char>((file_size >> 24) & 0xff)),

      0x00, 0x00, 0x00, 0x00,  // reserved
      0x36, 0x00, 0x00, 0x00,  // offset of start of image data
      0x28, 0x00, 0x00, 0x00,  // size of the DIB header

      // width in pixels
      (static_cast<unsigned char>(width & 0xff)),
      (static_cast<unsigned char>((width >> 8) & 0xff)),
      (static_cast<unsigned char>((width >> 16) & 0xff)),
      (static_cast<unsigned char>((width >> 24) & 0xff)),

      // height in pixels
      (static_cast<unsigned char>(height & 0xff)),
      (static_cast<unsigned char>((height >> 8) & 0xff)),
      (static_cast<unsigned char>((height >> 16) & 0xff)),
      (static_cast<unsigned char>((height >> 24) & 0xff)),

      0x01, 0x00,              // number of color planes
      0x18, 0x00,              // number of bits per pixel
      0x00, 0x00, 0x00, 0x00,  // no compression - BI_RGB
      0x00, 0x00, 0x00, 0x00,  // image size - dummy 0 for BI_RGB
      0x13, 0x0b, 0x00, 0x00,  // horizontal ppm
      0x13, 0x0b, 0x00, 0x00,  // vertical ppm
      0x00, 0x00, 0x00, 0x00,  // default 2^n colors in palette
      0x00, 0x00, 0x00, 0x00   // every color is important
  };
  // Open file for write
  std::ofstream output_bmp;
  output_bmp.open(file_path);
  if (!output_bmp) {
    error_code |= BmpError::INVALID_FILE;
    std::cerr << "ERROR: output file " << file_path << " does not exist."
              << std::endl;
    return;
  }

  // Write header
  output_bmp.write(reinterpret_cast<char *>(header), BMP_HEADER_SIZE);
  bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
  if (write_err) {
    error_code |= BmpError::FILE_IO_ERROR;
    std::cerr << "ERROR: could not write header to " << file_path << std::endl;
    return;
  }

  // Write data: Line size must be a multiple of 4 bytes
  int padding = (4 - ((width * 3) & 3)) & 3;
  for (int i = 0; i < height; ++i) {
    unsigned char p[3];
    for (int j = 0; j < width; ++j) {
      // written in B, G, R order
      p[0] = (bitmap_rgb(i, j) >> 0) & 0xff;   // B
      p[1] = (bitmap_rgb(i, j) >> 8) & 0xff;   // G
      p[2] = (bitmap_rgb(i, j) >> 16) & 0xff;  // R

      output_bmp.write(reinterpret_cast<char *>(p), 3);
      bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
      if (write_err) {
        write_err |= BmpError::FILE_IO_ERROR;
        std::cerr << "ERROR: could not write data to " << file_path
                  << std::endl;
        return;
      }
    }
    // Pad to multiple of 4 bytes
    if (padding) {
      p[0] = p[1] = p[2] = 0;
      output_bmp.write(reinterpret_cast<char *>(p), 3);
      bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
      if (write_err) {
        write_err |= BmpError::FILE_IO_ERROR;
        std::cerr << "ERROR: could not write padding to " << file_path
                  << std::endl;
        return;
      }
    }
  }
  output_bmp.close();
  return;
}

/// @brief Compare a `BitmapRGB` with a BMP file
/// @param bitmap_rgb A bitmap image to compare.
/// @param expectedFilePath Path to a BMP file to compare `bitmap_rgb` against.
/// @return `true` if `bitmap_rgb` matches the file pointed to by
/// `expectedFilePath`.
bool CompareFrames(BitmapRGB bitmap_rgb, std::string &expectedFilePath,
                   unsigned int error_code, unsigned char threshold = 2) {
  bool passed = false;
  error_code = BmpError::OK;
  unsigned int read_error;
  BitmapRGB bitmap_expected = bmp_tools::ReadBmp(expectedFilePath, read_error);
  if (read_error == BmpError::OK) {
    size_t rows = bitmap_rgb.GetRows();
    size_t cols = bitmap_rgb.GetCols();
    size_t exp_rows = bitmap_expected.GetRows();
    size_t exp_cols = bitmap_expected.GetCols();

    // check dimensions
    bool dims_ok = (rows == exp_rows) && (cols == exp_cols);
    if (!dims_ok) {
      error_code |= BmpError::INVALID_DIMENSIONS;
      std::cerr << "ERROR: output dimensions (" << cols << ", " << rows
                << ") do not match expected dimensions (" << exp_cols << ", "
                << exp_rows << ")." << std::endl;
      return false;
    }

    passed = dims_ok;
    // compare image data
    if (passed) {
      for (int i = 0; i < (rows * cols); ++i) {
        // Allow for some error due to fpc and fp-relaxed
        passed &= PixelRGB(bitmap_rgb(i))
                      .CheckSimilarity(PixelRGB(bitmap_expected(i)), threshold);
      }
    }
  } else {
    error_code |= BmpError::FILE_IO_ERROR;
    std::cerr << "ERROR: problem reading expected image " << expectedFilePath
              << std::endl;
    return false;
  }
  return passed;
}

}  // namespace bmp_tools

#endif  // BMP_TOOLS_H
