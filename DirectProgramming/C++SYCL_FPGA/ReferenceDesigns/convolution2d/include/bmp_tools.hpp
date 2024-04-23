//  Copyright (c) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT

// bmp_tools.h

// This header file parses bitmap image files.

#ifndef BMP_TOOLS_H
#define BMP_TOOLS_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>

#ifndef FILENAME_BUF_SIZE
#if defined(_WIN32) || defined(_WIN64)
#define FILENAME_BUF_SIZE _MAX_PATH
#else
#define FILENAME_BUF_SIZE MAX_PATH
#endif
#endif

// HLS did not allow exception handling, but oneAPI does.

// #ifndef ENABLE_EXCEPTION_HANDLING
// #define ENABLE_EXCEPTION_HANDLING 0
// #endif

namespace bmp_tools {

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
  uint16_t img_bitdepth;
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

/// @brief Read a `.bmp` file pointed to by `filename` into memory. The image
/// pixels will be stored as 32-bit unsigned integers. This function will
/// allocate memory and store a pointer to that memory in `img_data`.
///
/// @paragraph For simplicity, we only support a certain type of BMP file,
/// namely 24-bit Windows-style, with all important colors and a single color
/// plane.
///
/// @param[in] file_path File path to read from
///
/// @param[out] img_data A pointer to an array of pixels that has been allocated
/// by `ReadBmp`. The size of the allocated buffer will be equal to `height` *
/// `width` * `sizeof(unsigned int)`. Declare `unsigned int *myData`, then pass
/// `&myData` as an argument to this function.
///
/// @param[out] height Number of rows in the image that was read
///
/// @param[out] width Number of columns in the image that was read
///
/// @return `true` if succeeded, `false` if failed.
inline bool ReadBmp(std::string &file_path, unsigned int **img_data,
                     int &height, int &width) {
  if (nullptr == img_data) {
    std::cerr << "ERROR: img_data must not be null." << std::endl;
    return false;
  }
  std::ifstream input_bmp;
#if ENABLE_EXCEPTION_HANDLING
  try {
#endif
    input_bmp.open(file_path, std::ios::in | std::ios::binary);
    if (!input_bmp) {
      std::cerr << "ERROR: input file " << file_path << " does not exist."
                << std::endl;
      return false;
    }

#if ENABLE_EXCEPTION_HANDLING
  } catch (std::ios_base::failure &e) {
    std::cerr << e.what() << '\n';
    std::cerr << "ERROR: can't open file " << file_path << " for binary reading"
              << std::endl;
    return false;
  }
#endif

  bool failed = false;

  // load file header
  BmpFileHeader file_header;
  input_bmp.read(reinterpret_cast<char *>(&file_header), sizeof(BmpFileHeader));

  if (file_header.header_field != BmpHeaderField::BM) {
    std::cerr << "ERROR: only Windows-format bitmap header is supported. "
                 "Please convert to a Windows-style bitmap."
              << std::endl;
    return false;
  }

  WindowsBitmapInfoHeader dib_header;
  input_bmp.read(reinterpret_cast<char *>(&dib_header),
                sizeof(WindowsBitmapInfoHeader));

  width = dib_header.img_width;
  height = dib_header.img_height;

  // sanity check that inputs are valid
  if (dib_header.img_bitdepth != 24) {
    std::cerr << "ERROR: Only 24-bit BMP is supported. Please ensure your BMP "
                 "uses 24-bit pixels (24)"
              << std::endl;
    return false;
  }

  if (dib_header.img_planes != 1) {
    std::cerr << "ERROR: Only 1-plane BMP is supported. Please ensure your BMP "
                 "uses a single color plane (1)."
              << std::endl;
    return false;
  }

  if (dib_header.img_colors != 0) {
    std::cerr << "ERROR: requires 2^n colors. Please ensure your BMP uses the "
                 "default number of colors (0)."
              << std::endl;
    return false;
  }

  if (dib_header.img_important_colors != 0) {
    std::cerr
        << "ERROR: all colors should be important. Please ensure your BMP uses "
           "the default number of important colors (0)."
        << std::endl;
    return false;
  }

  // check that width*height is also valid
  if ((height < 0 || width < 0) || (height * width < 0)) {
    std::cerr << "ERROR: got height " << height << ", width " << width
              << std::endl;
    return false;
  }
  *img_data = (unsigned int *)malloc(width * height * sizeof(unsigned int));

  if (!*img_data) {
    std::cerr << "ERROR: Failed to allocate memory for img_data." << std::endl;
    return false;
  }

  // scroll to image data
  auto read_bytes = sizeof(WindowsBitmapInfoHeader) + sizeof(BmpFileHeader);
  while (read_bytes < file_header.img_data_offset) {
    input_bmp.get();
    read_bytes++;
  }

  // BMP: Each line must be a multiple of 4 bytes
  int padding = (4 - ((width * 3) & 3)) & 3;
  int idx = 0;

  // Color order is BGR, read across bottom row, then repeat going up rows
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      unsigned char b = input_bmp.get();  // B
      unsigned char g = input_bmp.get();  // G

      std::ios_base::iostate state = input_bmp.rdstate();

      bool earlyEOF = state & std::ifstream::eofbit;
      unsigned char r = input_bmp.get();  // R
      (*img_data)[idx] = (((unsigned int)r << 16) | ((unsigned int)g << 8) |
                          ((unsigned int)b << 0));
      idx++;
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
    std::cerr << "ERROR: File I/O error" << std::endl;
    free(*img_data);
    return false;
  }

  return true;
}

/// @brief Store pixels in `img_data` array to a bitmap pointed to by `fileame`
/// @paragraph For simplicity, we only support a certain type of BMP file,
/// namely 24-bit Windows-style, with all important colors and a single color
/// plane.
/// @param[in] file_path Filepath to write to
/// @param[in] img_data An array of pixels to write to a bmp file. The size of
/// the array must be equal to `height` * `width`.
/// @param[in] height Number of rows in the image
/// @param[in] width Number of columns in the image
/// @return `true` if the image was successfully written to the filesystem,
/// `false` if not.
inline bool WriteBmp(std::string &file_path, unsigned int *img_data, int height,
                      int width) {
  // sanity check that inputs are valid, check that width*height is also valid
  if ((height < 0 || width < 0) || (height * width > (1 << 30))) {
    std::cerr << "ERROR: height " << height << ", width " << width << std::endl;
    return false;
  }

  unsigned int file_size = width * height * 3 + BMP_HEADER_SIZE;
  unsigned char header[BMP_HEADER_SIZE] = {
      0x42, 0x4d,  // BMP & DIB

      // size of file in bytes
      (static_cast<unsigned char>(file_size & 0xff)),
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
      0x00, 0x00, 0x00, 0x00,  // default 2^n colors in palatte
      0x00, 0x00, 0x00, 0x00   // every color is important
  };
  // Open file for write
  std::ofstream output_bmp;
#if ENABLE_EXCEPTION_HANDLING
  try {
#endif
    output_bmp.open(file_path);
    if (!output_bmp) {
      std::cerr << "ERROR: output file " << file_path << " does not exist."
                << std::endl;
      return false;
    }
#if ENABLE_EXCEPTION_HANDLING
  } catch (std::ios_base::failure &e) {
    std::cerr << e.what() << '\n';
    std::cerr << "ERROR: can't open file " << file_path << " for binary writing"
              << std::endl;
    return false;
  }
#endif

  // Write header
  output_bmp.write(reinterpret_cast<char *>(header), BMP_HEADER_SIZE);
  bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
  if (write_err) {
    std::cerr << "ERROR: could not write header to " << file_path << std::endl;
    return false;
  }

  // Write data: Line size must be a multiple of 4 bytes
  int padding = (4 - ((width * 3) & 3)) & 3;
  unsigned int idx = 0;
  for (int i = 0; i < height; ++i) {
    unsigned char p[3];
    for (int j = 0; j < width; ++j) {
      // written in B, G, R order
      p[0] = (img_data[idx] >> 0) & 0xff;   // B
      p[1] = (img_data[idx] >> 8) & 0xff;   // G
      p[2] = (img_data[idx] >> 16) & 0xff;  // R
      idx++;

      output_bmp.write(reinterpret_cast<char *>(p), 3);
      bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
      if (write_err) {
        std::cerr << "ERROR: could not write data to " << file_path << std::endl;
        return false;
      }
    }
    // Pad to multiple of 4 bytes
    if (padding) {
      p[0] = p[1] = p[2] = 0;
      output_bmp.write(reinterpret_cast<char *>(p), 3);
      bool write_err = (output_bmp.rdstate() != std::ofstream::goodbit);
      if (write_err) {
        std::cerr << "ERROR: could not write padding to " << file_path
                  << std::endl;
        return false;
      }
    }
  }
  output_bmp.close();
  return true;
}

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

  /// @brief Transform a `bmp_tools::PixelRGB` into an unsigned 32-bit pixel used
  /// by `bmp_tools` functions.
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

/// @brief Compare an array of pixels with a BMP file
/// @param frame An array of unsigned integers representing pixels as understood
/// by the bitmap processing functions in `bmp_tools.hpp`.
/// @param rows Image height
/// @param cols Image width
/// @param expectedFilePath Path to a BMP file to compare `frame` against. The
/// BMP file will be parsed using the functions in `bmp_tools.hpp`.
/// @return `true` if `frame` matches the file pointed to by `expectedFilePath
 bool CompareFrames(unsigned int *frame, int rows, int cols,
                   std::string &expectedFilePath, unsigned char threshold = 2) {
  unsigned int *exp_img = nullptr;
  int exp_rows, exp_cols;
  bool passed = false;
  if (bmp_tools::ReadBmp(expectedFilePath, &exp_img, exp_rows, exp_cols)) {
    // check dimensions
    bool dims_ok = (rows == exp_rows) && (cols == exp_cols);
    if (!dims_ok) {
      std::cerr << "ERROR: output dimensions (" << cols << ", " << rows
                << ") do not match expected dimensions (" << exp_cols << ", "
                << exp_rows << ")." << std::endl;
    }

    bool frame_ok = (nullptr != frame);
    if (!frame_ok) {
      std::cerr << "ERROR: frame pointer invalid." << std::endl;
    }

    bool exp_ok = (nullptr != exp_img);
    if (!exp_ok) {
      std::cerr << "ERROR: exp_img pointer invalid." << std::endl;
    }
    bool pointers_ok = exp_ok & frame_ok;

    passed = dims_ok & pointers_ok;
    // compare image data
    if (passed) {
      for (int i = 0; i < (rows * cols); ++i) {
        // Allow for some error due to fpc and fp-relaxed
        passed &= PixelRGB(frame[i]).CheckSimilarity(PixelRGB(exp_img[i]),
                                                      threshold);
      }
    }
  } else {
    std::cerr << "ERROR: problem reading expected image " << expectedFilePath
              << std::endl;
  }
  free(exp_img);
  return passed;
}

}  // namespace bmp_tools

#endif  // BMP_TOOLS_H
