
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef USE_GPU
#include "gpu.h"
#endif

#include "create_voxels.h"
#include "framebuffer.h"

#include <openvkl/openvkl.h>

#include <openvkl/device/openvkl.h>

int main(int argc, char **argv)
{
#ifdef USE_GPU
  sycl::queue syclQueue = initSyclQueue();
#endif

  vklInit();

#ifndef USE_GPU
  VKLDevice device = vklNewDevice("cpu");
#else
  VKLDevice device          = vklNewDevice("gpu");
  sycl::context syclContext = syclQueue.get_context();

  vklDeviceSetVoidPtr(device, "syclContext", static_cast<void *>(&syclContext));
#endif

  vklCommitDevice(device);

  constexpr size_t res      = 128;
  std::vector<float> voxels = createVoxels(res);

  VKLVolume volume = vklNewVolume(device, "structuredRegular");
  vklSetVec3i(volume, "dimensions", res, res, res);

  const float spacing = 1.f / static_cast<float>(res);
  vklSetVec3f(volume, "gridSpacing", spacing, spacing, spacing);
  VKLData voxelData =
      vklNewData(device, voxels.size(), VKL_FLOAT, voxels.data());
  vklSetData(volume, "data", voxelData);
  vklRelease(voxelData);

  vklCommit(volume);

  VKLSampler sampler = vklNewSampler(volume);
  vklCommit(sampler);

#ifdef USE_GPU
  Framebuffer<AllocatorSycl<Pixel>> fb(64, 32, syclQueue);
#else
  Framebuffer<> fb(64, 32);
#endif

  // We trace the volume with simple ray marching.
  // Conceptually, this is a series of camera-aligned,
  // semi transparent planes.
  // We walk along the ray in regular steps.
  const int numSteps = 8;
  const float tMax   = 1.f;
  const float tStep  = tMax / numSteps;
  fb.generate([=](float fx, float fy) {
    Color color = {0.f};
    for (int i = 0; i < numSteps; ++i) {
      const vkl_vec3f p = {fx, fy, i * tStep};
      const Color c     = transferFunction(vklComputeSample(&sampler, &p));

      // We use the over operator to blend semi-transparent
      // "surfaces" together.
      color = over(color, c);

      // Now we've created a very simple volume renderer using
      // Open VKL!
    }
    return color;
  });

  fb.drawToTerminal();

  vklRelease(sampler);
  vklRelease(volume);
  vklReleaseDevice(device);

  return 0;
}
