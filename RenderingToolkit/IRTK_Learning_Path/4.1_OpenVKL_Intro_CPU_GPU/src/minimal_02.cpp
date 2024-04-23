
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef USE_GPU
#include "gpu.h"
#endif

#include "framebuffer.h"

// We must include the openvkl header.
#include <openvkl/openvkl.h>

#include <openvkl/device/openvkl.h>

int main(int argc, char **argv)
{
#ifdef USE_GPU
  sycl::queue syclQueue = initSyclQueue();
#endif

  // To initialize Open VKL, load the device module, which is essentially the
  // backend implementation. Our current release supports a "cpu" device
  // which is highly optimized for vector CPU architectures, and a "gpu" device
  // optimized for GPUs.
  vklInit();

#ifndef USE_GPU
  // The device itself will be manage all resources. cpu selects the native
  // vector width for best performance.
  VKLDevice device = vklNewDevice("cpu");
#else
  // For GPU, we need to provide a SYCL context.
  VKLDevice device          = vklNewDevice("gpu");
  sycl::context syclContext = syclQueue.get_context();

  vklDeviceSetVoidPtr(device, "syclContext", static_cast<void *>(&syclContext));
#endif

  // Devices must be committed before use. This is because they support
  // parameters, such as logging verbosity.
  vklCommitDevice(device);

#ifdef USE_GPU
  // debug: see if this resolves link errors in GPU device
  VKLVolume volume = vklNewVolume(device, "structuredRegular");
  vklCommit(volume);
#endif

#ifdef USE_GPU
  Framebuffer<AllocatorSycl<Pixel>> fb(64, 32, syclQueue);
#else
  Framebuffer<> fb(64, 32);
#endif

  fb.generate([=](float fx, float fy) { return transferFunction(2 * fx - 1); });
  fb.drawToTerminal();

  // When the application is done with the device, release it!
  // This will clean up the internal state.
  vklReleaseDevice(device);

  return 0;
}
