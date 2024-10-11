
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef USE_GPU
#include "gpu.h"
#endif

#include "framebuffer.h"

int main(int argc, char **argv)
{
#ifdef USE_GPU
  // on GPU we need to create a SYCL queue.
  sycl::queue syclQueue = initSyclQueue();
#endif

#ifdef USE_GPU
  // on GPU, we provide the SYCL queue to facilitate GPU memory allocations.
  Framebuffer<AllocatorSycl<Pixel>> fb(64, 32, syclQueue);
#else
  Framebuffer<> fb(64, 32);
#endif

  fb.generate([=](float fx, float fy) { return transferFunction(2 * fx - 1); });
  fb.drawToTerminal();

  return 0;
}
