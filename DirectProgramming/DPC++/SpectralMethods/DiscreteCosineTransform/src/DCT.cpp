//=======================================================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// ======================================================================================

#include "DCT.hpp"

#include <CL/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "dpc_common.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

using namespace dpc_common;
using namespace sycl;

constexpr int num_tests = 5;

constexpr int block_dims = 8;
constexpr int block_size = 64;

// API for creating 8x8 DCT matrix
void CreateDCT(float matrix[block_size]) {
  int temp[block_dims];
  for (int i = 0; i < block_dims; ++i) temp[i] = i;
  for (int i = 0; i < block_dims; ++i) {
    for (int j = 0; j < block_dims; ++j) {
      if (i == 0)
        matrix[(i * block_dims) + j] = (1 / sycl::sqrt((float)block_dims));
      else
        matrix[(i * block_dims) + j] =
            sycl::sqrt((float)2 / block_dims) *
            sycl::cos(((((float)2 * temp[j]) + 1) * i * 3.14f) /
                      (2 * block_dims));
    }
  }
}

// Transposes an 8x8 matrix x and writes output to xinv
void MatrixTranspose(float x[block_size], float xinv[block_size]) {
  for (int i = 0; i < block_dims; ++i) {
    for (int j = 0; j < block_dims; ++j)
      xinv[(j * block_dims) + i] = x[(i * block_dims) + j];
  }
}

// Multiply two matrices x and y and write output to xy
SYCL_EXTERNAL void MatrixMultiply(float x[block_size], float y[block_size],
                                  float xy[block_size]) {
  for (int i = 0; i < block_dims; ++i) {
    for (int j = 0; j < block_dims; ++j) {
      xy[(i * block_dims) + j] = 0;
      for (int k = 0; k < block_dims; ++k)
        xy[(i * block_dims) + j] +=
            (x[(i * block_dims) + k] * y[(k * block_dims) + j]);
    }
  }
}

// Processes an individual 8x8 subset of image data
SYCL_EXTERNAL void ProcessBlock(rgb* indataset, rgb* outdataset,
                                float dct[block_size], float dctinv[block_size],
                                int start_index, int width) {
  float interim[block_size], product[block_size], red_input[block_size],
      blue_input[block_size], green_input[block_size], temp[block_size];

  /*
  // Quantization matrix which does 50% quantization
  float quant[64] = {16, 11, 10, 16, 24,  40,  51,  61,
                     12, 12, 14, 19, 26,  58,  60,  55,
                     14, 13, 16, 24, 40,  57,  69,  56,
                     14, 17, 22, 29, 51,  87,  80,  62,
                     18, 22, 37, 56, 68,  109, 103, 77,
                     24, 35, 55, 64, 81,  104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101,
                     72, 92, 95, 98, 112, 100, 103, 99 };
  */
  // Quantization matrix which does 90% quantization
  float quant[64] = {3,  2,  2,  3,  5,  8,  10, 12,
                     2,  2,  3,  4,  5,  12, 12, 11,
                     3,  3,  3,  5,  8,  11, 14, 11,
                     3,  3,  4,  6,  10, 17, 16, 12,
                     4,  4,  7,  11, 14, 22, 21, 15,
                     5,  7,  11, 13, 16, 12, 23, 18,
                     10, 13, 16, 17, 21, 24, 24, 21,
                     14, 18, 19, 20, 22, 20, 20, 20};
  /*
  // Quantization matrix which does 10% quantization
  float quant[64] = {80,  60,  50,  80,  120, 200, 255, 255,
                     55,  60,  70,  95,  130, 255, 255, 255,
                     70,  65,  80,  120, 200, 255, 255, 255,
                     70,  85,  110, 145, 255, 255, 255, 255,
                     90,  110, 185, 255, 255, 255, 255, 255,
                     120, 175, 255, 255, 255, 255, 255, 255,
                     245, 255, 255, 255, 255, 255, 255, 255,
                     255, 255, 255, 255, 255, 255, 255, 255 };
  */

  // PROCESS RED CHANNEL

  // Translating the pixels values from [0, 255] range to [-128, 127] range
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    red_input[i] = indataset[start_index + pixel_index].red;
    red_input[i] -= 128;
  }

  // Computation of the discrete cosine transform of the image section of size
  // 8x8 for red values
  MatrixMultiply(dct, red_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  // Computation of quantization phase using the quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // Computation of dequantizing phase using the same above quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // Computation of Inverse Discrete Cosine Transform (IDCT)
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // Translating the pixels values from [-128, 127] range to [0, 255] range
  // and writing to output image data
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = (product[i] + 128);
    outdataset[start_index + pixel_index].red =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }

  // PROCESS BLUE CHANNEL

  // Translating the pixels values from [0, 255] range to [-128, 127] range
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    blue_input[i] = indataset[start_index + pixel_index].blue;
    blue_input[i] -= 128;
  }

  // Computation of the discrete cosine transform of the image section of size
  // 8x8 for blue values
  MatrixMultiply(dct, blue_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  // Computation of quantization phase using the quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // Computation of dequantizing phase using the same above quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // Computation of Inverse Discrete Cosine Transform (IDCT)
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // Translating the pixels values from [-128, 127] range to [0, 255] range
  // and writing to output image data
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = product[i] + 128;
    outdataset[start_index + pixel_index].blue =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }

  // PROCESS GREEN CHANNEL

  // Translating the pixels values from [0, 255] range to [-128, 127] range
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    green_input[i] = indataset[start_index + pixel_index].green;
    green_input[i] -= 128;
  }

  // Computation of the discrete cosine transform of the image section of size
  // 8x8 for green values
  MatrixMultiply(dct, green_input, temp);
  MatrixMultiply(temp, dctinv, interim);

  // Computation of quantization phase using the quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] / quant[i]) + 0.5f);

  // Computation of dequantizing phase using the same above quantization matrix
  for (int i = 0; i < block_size; ++i)
    interim[i] = sycl::floor((interim[i] * quant[i]) + 0.5f);

  // Computation of Inverse Discrete Cosine Transform (IDCT)
  MatrixMultiply(dctinv, interim, temp);
  MatrixMultiply(temp, dct, product);

  // Translating the pixels values from [-128, 127] range to [0, 255] range
  // and writing to output image data
  for (int i = 0; i < block_size; ++i) {
    int pixel_index = i / block_dims * width + i % block_dims;
    float temp = product[i] + 128;
    outdataset[start_index + pixel_index].green =
        (temp > 255.f) ? 255 : (unsigned char)temp;
  }
}

// Breaks the image into 8x8 blocks to process DCT
void ProcessImage(rgb* indataset, rgb* outdataset, int width, int height) {
  sycl::queue q(default_selector{}, exception_handler);
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  try {
    int image_size = width * height;
    float dct[block_size], dctinv[block_size];

    // Creation of 8x8 DCT matrix
    CreateDCT(dct);
    // Creating a transpose of DCT matrix
    MatrixTranspose(dct, dctinv);

    buffer indata_buf(indataset, range<1>(image_size));
    buffer outdata_buf(outdataset, range<1>(image_size));
    buffer dct_buf(dct, range<1>(block_size));
    buffer dctinv_buf(dctinv, range<1>(block_size));

    q.submit([&](handler& h) {
      auto i_acc = indata_buf.get_access(h,read_only);
      auto o_acc = outdata_buf.get_access(h);
      auto d_acc = dct_buf.get_access(h,read_only);
      auto di_acc = dctinv_buf.get_access(h,read_only);

      // Processes individual 8x8 chunks in parallel
      h.parallel_for(
          range<2>(width / block_dims, height / block_dims), [=](auto idx) {
            int start_index = idx[0] * block_dims + idx[1] * block_dims * width;
            ProcessBlock(i_acc.get_pointer(), o_acc.get_pointer(),
                         d_acc.get_pointer(), di_acc.get_pointer(), start_index,
                         width);
          });
    });
    q.wait_and_throw();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what() << "\n";
    exit(1);
  }
}

// This API does the reading and writing from/to the .bmp file. Also invokes the
// image processing API from here
int ReadProcessWrite(char* input, char* output) {
  double timersecs;
#ifdef PERF_NUM
  double avg_timersecs = 0;
#endif

  // Read in the data from the input image file
  int image_width = 0, image_height = 0, num_channels = 0;
  rgb* indata = (rgb*)stbi_load(input, &image_width, &image_height,
                                &num_channels, STBI_rgb);

  if (!indata) {
    std::cout << "The input file could not be opened. Program will now exit\n";
    return 1;
  } else if (num_channels != 3) {
    std::cout
        << "The input file must be an RGB bmp image. Program will now exit\n";
    return 1;
  } else if (image_width % block_dims != 0 || image_height % block_dims != 0) {
    std::cout
        << "The input image must have dimensions which are a multiple of 8\n";
    return 1;
  }

  std::cout << "Filename: " << input << " W: " << image_width
            << " H: " << image_height << "\n\n";

  rgb* outdata = (rgb*)malloc(image_width * image_height * sizeof(rgb));

  // Invoking the DCT/Quantization API which does some manipulation on the
  // bitmap data read from the input .bmp file
#ifdef PERF_NUM
  std::cout << "Run all tests...\n\n";
  for (int j = 0; j < num_tests; ++j) {
#endif
    std::cout << "Start image processing with offloading to GPU...\n";
    {
      TimeInterval t;
      ProcessImage(indata, outdata, image_width, image_height);
      timersecs = t.Elapsed();
    }
    std::cout << "--The processing time is " << timersecs << " seconds\n\n";
#ifdef PERF_NUM
    avg_timersecs += timersecs;
  }
#endif

  stbi_write_bmp(output, image_width, image_height, 3, outdata);
  std::cout << "DCT successfully completed on the device.\n"
               "The processed image has been written to " << output << "\n";

#ifdef PERF_NUM
  std::cout << "\nAverage time for image processing:\n";
  std::cout << "--The average processing time was "
            << avg_timersecs / (float)num_tests << " seconds\n";
#endif
  // Freeing dynamically allocated memory
  stbi_image_free(indata);
  std::free(outdata);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Program usage is <modified_program> <inputfile.bmp> "
                 "<outputfile.bmp>\n";
    return 1;
  }
  return ReadProcessWrite(argv[1], argv[2]);
}
