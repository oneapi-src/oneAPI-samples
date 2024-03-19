
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef USE_GPU
#include "gpu.h"
#endif

#include "create_voxels.h"
#include "framebuffer.h"

// We must include the openvkl header.
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

  // "Load data from disk". (We generate the array procedurally).
  constexpr size_t res      = 128;
  std::vector<float> voxels = createVoxels(res);

  // Note that Open VKL uses a C99 API for maximum compatibility.
  // So we will have to wrap the array we just created so that
  // we can pass it to Open VKL.

  // Create a new volume. Volume objects are created on a device.
  // We create a structured regular grid here, which is essentially
  // a dense 3D array.
  VKLVolume volume = vklNewVolume(device, "structuredRegular");

  // We have to set a few parameters on the volume.
  // First, Open VKL needs to know the extent of the volume:
  vklSetVec3i(volume, "dimensions", res, res, res);

  // By default, the volume assumes a voxel size of 1. Scale it so the
  // domain is [0, 1].
  const float spacing = 1.f / static_cast<float>(res);
  vklSetVec3f(volume, "gridSpacing", spacing, spacing, spacing);

  // Open VKL has a concept of typed Data objects. That's how we pass data
  // buffers to a device.
  VKLData voxelData =
      vklNewData(device, voxels.size(), VKL_FLOAT, voxels.data());

  // Set the data parameter. We can release the data directly afterwards
  // as Open VKL has a reference counting mechanism and will keep track
  // internally.
  vklSetData(volume, "data", voxelData);
  vklRelease(voxelData);

  // Finally, commit. This may build acceleration structures, etc.
  vklCommit(volume);

  // Instead of drawing the field directly into our framebuffer, we will instead
  // sample the volume we just created. To do that, we need a sampler object.
  VKLSampler sampler = vklNewSampler(volume);
  vklCommit(sampler);

#ifdef USE_GPU
  Framebuffer<AllocatorSycl<Pixel>> fb(64, 32, syclQueue);
#else
  Framebuffer<> fb(64, 32);
#endif

  fb.generate([=](float fx, float fy) {
    // To sample, we call vklComputeSample on our sampler object.
    const vkl_vec3f p = {fx, fy, 0.f};
    return transferFunction(vklComputeSample(&sampler, &p));
  });

  fb.drawToTerminal();

  // Release the volume to clean up!
  vklRelease(sampler);
  vklRelease(volume);
  vklReleaseDevice(device);

  return 0;
}
