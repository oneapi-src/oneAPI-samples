
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

  // One advantage of Open VKL is that we can use a different data structure
  // with the same sampling API.
  // Here, we replace our data structure with a structured spherical volume
  // for a spherical domain.
  VKLVolume volume = vklNewVolume(device, "structuredSpherical");

  vklSetVec3i(volume, "dimensions", res, res, res);
  const float spacing = 1.f / static_cast<float>(res);
  // We must adapt gridSpacing, as structuredSpherical expects spacing
  // in spherical coordinates.
  vklSetVec3f(volume, "gridSpacing", spacing, 180.f * spacing, 360.f * spacing);

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

  fb.generate([=](float fx, float fy) {
    // Also try slice 1.0 to demonstrate a different view.
    const vkl_vec3f p = {fx, fy, 0.f};
    return transferFunction(vklComputeSample(&sampler, &p));
  });

  fb.drawToTerminal();

  vklRelease(sampler);
  vklRelease(volume);
  vklReleaseDevice(device);

  return 0;
}
