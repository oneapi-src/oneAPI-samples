/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  FFT-based Ocean simulation
  based on original code by Yury Uralsky and Calvin Lin

  This sample demonstrates how to use CUFFT to synthesize and
  render an ocean surface in real-time.

  See Jerry Tessendorf's Siggraph course notes for more details:
  http://tessendorf.org/reports.html

  It also serves as an example of how to generate multiple vertex
  buffer streams from CUDA and render them using GLSL shaders.
*/

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dpct/dpct.hpp>
#include <dpct/fft_utils.hpp>
#include <sycl/sycl.hpp>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

const char *sSDKsample = "SYCL FFT Ocean Simulation";

#define SYCLRT_SQRT_HALF_F 0.707106781f
#define MAX_EPSILON 0.10f
#define THRESHOLD 0.15f
#define REFRESH_DELAY 10  // ms

////////////////////////////////////////////////////////////////////////////////
// constants
unsigned int windowW = 512, windowH = 512;

const unsigned int meshSize = 256;
const unsigned int spectrumW = meshSize + 4;
const unsigned int spectrumH = meshSize + 1;

bool animate = true;

// FFT data
dpct::fft::fft_engine_ptr fftPlan;
sycl::float2 *d_h0 = 0;  // heightfield at time 0
sycl::float2 *h_h0 = 0;
sycl::float2 *d_ht = 0;  // heightfield at time t
sycl::float2 *d_slope = 0;

// pointers to device object
float *g_hptr = NULL;
sycl::float2 *g_sptr = NULL;

// simulation parameters
const float g = 9.81f;        // gravitational constant
const float A = 1e-7f;        // wave scale factor
const float patchSize = 100;  // patch size
float windSpeed = 100.0f;
float windDir = 3.141592654F / 3.0f;
float dirDepend = 0.07f;

StopWatchInterface *timer = NULL;
float animTime = 0.0f;
float prevTime = 0.0f;
float animationRate = -0.001f;

// Auto-Verification Code
const int frameCheckNumber = 4;
int fpsCount = 0;  // FPS count for averaging
int fpsLimit = 1;  // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

////////////////////////////////////////////////////////////////////////////////
// kernels

extern "C" void cudaGenerateSpectrumKernel(sycl::float2 *d_h0,
                                           sycl::float2 *d_ht,
                                           unsigned int in_width,
                                           unsigned int out_width,
                                           unsigned int out_height,
                                           float animTime, float patchSize);

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap,
                                          sycl::float2 *d_ht,
                                          unsigned int width,
                                          unsigned int height, bool autoTest);

extern "C" void cudaCalculateSlopeKernel(float *h, sycl::float2 *slopeOut,
                                         unsigned int width,
                                         unsigned int height);

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void runAutoTest(int argc, char **argv);

// rendering callbacks
void timerEvent(int value);

// Cuda functionality
void runCudaTest(char *exec_path);
void generate_h0(sycl::float2 *h0);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  printf(
      "NOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n\n");

  // check for command line arguments
  if (checkCmdLineFlag(argc, (const char **)argv, "qatest")) {
    animate = false;
    fpsLimit = frameCheckNumber;
    runAutoTest(argc, argv);
  } /* else {
     printf(
         "[%s]\n\n"
         "Left mouse button          - rotate\n"
         "Middle mouse button        - pan\n"
         "Right mouse button         - zoom\n"
         "'w' key                    - toggle wireframe\n",
         sSDKsample);

     runGraphicsTest(argc, argv);
   }*/

  exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run test
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int argc, char **argv) {
  printf("%s Starting...\n\n", argv[0]);

  // Cuda init
  // int dev = findCudaDevice(argc, (const char **)argv);

  int dev = 0;
  DPCT_CHECK_ERROR(dev = dpct::dev_mgr::instance().current_device_id());

  dpct::device_info deviceProp;

  DPCT_CHECK_ERROR(
      (dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp),
       0));

  // checkCudaErrors(DPCT_CHECK_ERROR(
  //     dpct::dev_mgr::instance().get_device(dev).get_device_info(deviceProp)));

  printf("Compute capability %d.%d\n", deviceProp.get_major_version(),
         deviceProp.get_minor_version());

  // create FFT plan
  DPCT_CHECK_ERROR(
      fftPlan = dpct::fft::fft_engine::create(
          &dpct::get_default_queue(), meshSize, meshSize,
          dpct::fft::fft_type::complex_float_to_complex_float));

  // allocate memory
  int spectrumSize = spectrumW * spectrumH * sizeof(sycl::float2);
  
  DPCT_CHECK_ERROR(d_h0 = (sycl::float2 *)sycl::malloc_device(
                           spectrumSize, dpct::get_default_queue()));
  h_h0 = (sycl::float2 *)malloc(spectrumSize);
  generate_h0(h_h0);
  DPCT_CHECK_ERROR(
      dpct::get_default_queue().memcpy(d_h0, h_h0, spectrumSize).wait());

  int outputSize = meshSize * meshSize * sizeof(sycl::float2);
  DPCT_CHECK_ERROR(d_ht = (sycl::float2 *)sycl::malloc_device(
                                       outputSize, dpct::get_default_queue()));
  
  DPCT_CHECK_ERROR(d_slope = (sycl::float2 *)sycl::malloc_device(
                           outputSize, dpct::get_default_queue()));

  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  prevTime = sdkGetTimerValue(&timer);

  runCudaTest(argv[0]);
  printf("Processing time : %f (ms)\n", sdkGetTimerValue(&timer));
  
  DPCT_CHECK_ERROR(sycl::free(d_ht, dpct::get_default_queue()));
  
  DPCT_CHECK_ERROR(sycl::free(d_slope, dpct::get_default_queue()));
  
  DPCT_CHECK_ERROR(sycl::free(d_h0, dpct::get_default_queue()));
  DPCT_CHECK_ERROR(dpct::fft::fft_engine::destroy(fftPlan));
  free(h_h0);

  exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

float urand() { return rand() / (float)RAND_MAX; }

// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss() {
  float u1 = urand();
  float u2 = urand();

  if (u1 < 1e-6f) {
    u1 = 1e-6f;
  }

  return sqrtf(-2 * logf(u1)) * cosf(2 * 3.141592654F * u2);
}

// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// A - constant
float phillips(float Kx, float Ky, float Vdir, float V, float A,
               float dir_depend) {
  float k_squared = Kx * Kx + Ky * Ky;

  if (k_squared == 0.0f) {
    return 0.0f;
  }

  // largest possible wave from constant wind of velocity v
  float L = V * V / g;

  float k_x = Kx / sqrtf(k_squared);
  float k_y = Ky / sqrtf(k_squared);
  float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

  float phillips = A * expf(-1.0f / (k_squared * L * L)) /
                   (k_squared * k_squared) * w_dot_k * w_dot_k;

  // filter out waves moving opposite to wind
  if (w_dot_k < 0.0f) {
    phillips *= dir_depend;
  }

  // damp out waves with very small length w << l
  // float w = L / 10000;
  // phillips *= expf(-k_squared * w * w);

  return phillips;
}

// Generate base heightfield in frequency space
void generate_h0(sycl::float2 *h0) {
  for (unsigned int y = 0; y <= meshSize; y++) {
    for (unsigned int x = 0; x <= meshSize; x++) {
      float kx =
          (-(int)meshSize / 2.0f + x) * (2.0f * 3.141592654F / patchSize);
      float ky =
          (-(int)meshSize / 2.0f + y) * (2.0f * 3.141592654F / patchSize);

      float P = sqrtf(phillips(kx, ky, windDir, windSpeed, A, dirDepend));

      if (kx == 0.0f && ky == 0.0f) {
        P = 0.0f;
      }

      // float Er = urand()*2.0f-1.0f;
      // float Ei = urand()*2.0f-1.0f;
      float Er = gauss();
      float Ei = gauss();

      float h0_re = Er * P * SYCLRT_SQRT_HALF_F;
      float h0_im = Ei * P * SYCLRT_SQRT_HALF_F;

      int i = y * spectrumW + x;
      h0[i].x() = h0_re;
      h0[i].y() = h0_im;
    }
  }
}

void runCudaTest(char *exec_path) {
  
  DPCT_CHECK_ERROR(g_hptr = sycl::malloc_device<float>(
                           meshSize * meshSize, dpct::get_default_queue()));
  
  DPCT_CHECK_ERROR(g_sptr = sycl::malloc_device<sycl::float2>(
                           meshSize * meshSize, dpct::get_default_queue()));

  // generate wave spectrum in frequency domain
  cudaGenerateSpectrumKernel(d_h0, d_ht, spectrumW, meshSize, meshSize,
                             animTime, patchSize);

  // execute inverse FFT to convert to spatial domain
  
  DPCT_CHECK_ERROR((fftPlan->compute<sycl::float2, sycl::float2>(
          d_ht, d_ht, dpct::fft::fft_direction::backward)));

  // update heightmap values
  cudaUpdateHeightmapKernel(g_hptr, d_ht, meshSize, meshSize, true);

  {
    float *hptr = (float *)malloc(meshSize * meshSize * sizeof(float));
    dpct::get_default_queue()
        .memcpy((void *)hptr, (void *)g_hptr,
                meshSize * meshSize * sizeof(float))
        .wait();
    sdkDumpBin((void *)hptr, meshSize * meshSize * sizeof(float),
               "spatialDomain.bin");

    if (!sdkCompareBin2BinFloat("spatialDomain.bin", "ref_spatialDomain.bin",
                                meshSize * meshSize, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(hptr);
  }

  // calculate slope for shading
  cudaCalculateSlopeKernel(g_hptr, g_sptr, meshSize, meshSize);

  {
    sycl::float2 *sptr =
        (sycl::float2 *)malloc(meshSize * meshSize * sizeof(sycl::float2));
    dpct::get_default_queue()
        .memcpy((void *)sptr, (void *)g_sptr,
                meshSize * meshSize * sizeof(sycl::float2))
        .wait();
    sdkDumpBin(sptr, meshSize * meshSize * sizeof(sycl::float2),
               "slopeShading.bin");

    if (!sdkCompareBin2BinFloat("slopeShading.bin", "ref_slopeShading.bin",
                                meshSize * meshSize * 2, MAX_EPSILON, THRESHOLD,
                                exec_path)) {
      g_TotalErrors++;
    }

    free(sptr);
  }

  
   DPCT_CHECK_ERROR(sycl::free(g_hptr, dpct::get_default_queue()));
   DPCT_CHECK_ERROR(sycl::free(g_sptr, dpct::get_default_queue()));
}
