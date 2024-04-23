// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <openvkl/openvkl.h>
#include <openvkl/device/openvkl.h>

#include <iomanip>
#include <iostream>

// setup specialization constant for feature flags
static_assert(std::is_trivially_copyable<VKLFeatureFlags>::value);

constexpr static sycl::specialization_id<VKLFeatureFlags> samplerSpecId{
    VKL_FEATURE_FLAGS_DEFAULT};

void demoGpuAPI(sycl::queue &syclQueue, VKLDevice device, VKLVolume volume) {
  std::cout << "demo of GPU API" << std::endl;

  std::cout << std::fixed << std::setprecision(6);

  VKLSampler sampler = vklNewSampler(volume);
  vklCommit(sampler);

  // feature flags improve performance on GPU, as well as JIT times
  const VKLFeatureFlags requiredFeatures = vklGetFeatureFlags(sampler);

  // bounding box
  vkl_box3f bbox = vklGetBoundingBox(volume);
  std::cout << "\tbounding box" << std::endl;
  std::cout << "\t\tlower = " << bbox.lower.x << " " << bbox.lower.y << " "
            << bbox.lower.z << std::endl;
  std::cout << "\t\tupper = " << bbox.upper.x << " " << bbox.upper.y << " "
            << bbox.upper.z << std::endl
            << std::endl;

  // number of attributes
  unsigned int numAttributes = vklGetNumAttributes(volume);
  std::cout << "\tnum attributes = " << numAttributes << std::endl;

  // value range for all attributes
  for (unsigned int i = 0; i < numAttributes; i++) {
    vkl_range1f valueRange = vklGetValueRange(volume, i);
    std::cout << "\tvalue range (attribute " << i << ") = (" << valueRange.lower
              << " " << valueRange.upper << ")" << std::endl;
  }

  std::cout << std::endl << "\tsampling" << std::endl;

  // coordinate for sampling / gradients
  vkl_vec3f coord = {1.f, 2.f, 3.f};
  std::cout << "\n\tcoord = " << coord.x << " " << coord.y << " " << coord.z
            << std::endl
            << std::endl;

  // sample, gradient (first attribute)
  const unsigned int attributeIndex = 0;
  const float time = 0.f;

  // USM shared allocations, required when we want to pass results back from GPU
  float *sample = sycl::malloc_shared<float>(1, syclQueue);
  vkl_vec3f *grad = sycl::malloc_shared<vkl_vec3f>(1, syclQueue);

  syclQueue
      .submit([=](sycl::handler &cgh) {
        cgh.set_specialization_constant<samplerSpecId>(requiredFeatures);

        cgh.single_task([=](sycl::kernel_handler kh) {
          const VKLFeatureFlags featureFlags =
              kh.get_specialization_constant<samplerSpecId>();

          *sample = vklComputeSample(&sampler, &coord, attributeIndex, time,
                                     featureFlags);
          *grad = vklComputeGradient(&sampler, &coord, attributeIndex, time,
                                     featureFlags);
        });
      })
      .wait();

  std::cout << "\tsampling and gradient computation (first attribute)"
            << std::endl;
  std::cout << "\t\tsample = " << *sample << std::endl;
  std::cout << "\t\tgrad   = " << grad->x << " " << grad->y << " " << grad->z
            << std::endl
            << std::endl;

  sycl::free(sample, syclQueue);
  sycl::free(grad, syclQueue);

  // sample (multiple attributes)
  const unsigned int M = 3;
  const unsigned int attributeIndices[] = {0, 1, 2};

  float *samples = sycl::malloc_shared<float>(M, syclQueue);

  syclQueue
      .submit([=](sycl::handler &cgh) {
        cgh.set_specialization_constant<samplerSpecId>(requiredFeatures);

        cgh.single_task([=](sycl::kernel_handler kh) {
          const VKLFeatureFlags featureFlags =
              kh.get_specialization_constant<samplerSpecId>();

          vklComputeSampleM(&sampler, &coord, samples, M, attributeIndices,
                            time, featureFlags);
        });
      })
      .wait();

  std::cout << "\tsampling (multiple attributes)" << std::endl;
  std::cout << "\t\tsamples = " << samples[0] << " " << samples[1] << " "
            << samples[2] << std::endl;

  sycl::free(samples, syclQueue);

  // interval iterator context setup
  std::cout << std::endl << "\tinterval iteration" << std::endl << std::endl;

  std::vector<vkl_range1f> ranges{{10, 20}, {50, 75}};
  VKLData rangesData =
      vklNewData(device, ranges.size(), VKL_BOX1F, ranges.data());

  VKLIntervalIteratorContext intervalContext =
      vklNewIntervalIteratorContext(sampler);

  vklSetInt(intervalContext, "attributeIndex", 0);

  vklSetData(intervalContext, "valueRanges", rangesData);
  vklRelease(rangesData);

  vklCommit(intervalContext);

  // ray definition for iterators
  vkl_vec3f rayOrigin{0.f, 1.f, 1.f};
  vkl_vec3f rayDirection{1.f, 0.f, 0.f};
  vkl_range1f rayTRange{0.f, 200.f};
  std::cout << "\trayOrigin = " << rayOrigin.x << " " << rayOrigin.y << " "
            << rayOrigin.z << std::endl;
  std::cout << "\trayDirection = " << rayDirection.x << " " << rayDirection.y
            << " " << rayDirection.z << std::endl;
  std::cout << "\trayTRange = " << rayTRange.lower << " " << rayTRange.upper
            << std::endl
            << std::endl;

  // interval iteration
  char *iteratorBuffer = sycl::malloc_device<char>(
      vklGetIntervalIteratorSize(&intervalContext), syclQueue);

  int *numIntervals = sycl::malloc_shared<int>(1, syclQueue);
  *numIntervals = 0;

  const size_t maxNumIntervals = 999;

  VKLInterval *intervalsBuffer =
      sycl::malloc_shared<VKLInterval>(maxNumIntervals, syclQueue);
  memset(intervalsBuffer, 0, maxNumIntervals * sizeof(VKLInterval));

  std::cout << "\tinterval iterator for value ranges";

  for (const auto &r : ranges) {
    std::cout << " {" << r.lower << " " << r.upper << "}";
  }
  std::cout << std::endl << std::endl;

  syclQueue
      .submit([=](sycl::handler &cgh) {
        cgh.set_specialization_constant<samplerSpecId>(requiredFeatures);

        cgh.single_task([=](sycl::kernel_handler kh) {
          const VKLFeatureFlags featureFlags =
              kh.get_specialization_constant<samplerSpecId>();

          VKLIntervalIterator intervalIterator = vklInitIntervalIterator(
              &intervalContext, &rayOrigin, &rayDirection, &rayTRange, time,
              (void *)iteratorBuffer, featureFlags);

          for (;;) {
            VKLInterval interval;
            int result =
                vklIterateInterval(intervalIterator, &interval, featureFlags);
            if (!result) {
              break;
            }
            intervalsBuffer[*numIntervals] = interval;

            *numIntervals = *numIntervals + 1;
            if (*numIntervals >= maxNumIntervals) break;
          }
        });
      })
      .wait();

  for (int i = 0; i < *numIntervals; ++i) {
    std::cout << "\t\ttRange (" << intervalsBuffer[i].tRange.lower << " "
              << intervalsBuffer[i].tRange.upper << ")" << std::endl;
    std::cout << "\t\tvalueRange (" << intervalsBuffer[i].valueRange.lower
              << " " << intervalsBuffer[i].valueRange.upper << ")" << std::endl;
    std::cout << "\t\tnominalDeltaT " << intervalsBuffer[i].nominalDeltaT
              << std::endl
              << std::endl;
  }

  sycl::free(iteratorBuffer, syclQueue);
  sycl::free(numIntervals, syclQueue);
  sycl::free(intervalsBuffer, syclQueue);

  vklRelease(intervalContext);

  // hit iteration
  std::cout << std::endl << "\thit iteration" << std::endl << std::endl;

  // hit iterator context setup
  float values[2] = {32.f, 96.f};
  int num_values = 2;
  VKLData valuesData = vklNewData(device, num_values, VKL_FLOAT, values);

  VKLHitIteratorContext hitContext = vklNewHitIteratorContext(sampler);

  vklSetInt(hitContext, "attributeIndex", 0);

  vklSetData(hitContext, "values", valuesData);
  vklRelease(valuesData);

  vklCommit(hitContext);

  // ray definition for iterators
  // see rayOrigin, Direction and TRange above

  char *hitIteratorBuffer =
      sycl::malloc_device<char>(vklGetHitIteratorSize(&hitContext), syclQueue);

  int *numHits = sycl::malloc_shared<int>(1, syclQueue);
  *numHits = 0;

  const size_t maxNumHits = 999;

  VKLHit *hitBuffer = sycl::malloc_shared<VKLHit>(maxNumHits, syclQueue);
  memset(hitBuffer, 0, maxNumHits * sizeof(VKLHit));

  std::cout << "\thit iterator for values";

  for (const auto &r : values) {
    std::cout << " " << r << " ";
  }
  std::cout << std::endl << std::endl;

  syclQueue
      .submit([=](sycl::handler &cgh) {
        cgh.set_specialization_constant<samplerSpecId>(requiredFeatures);

        cgh.single_task([=](sycl::kernel_handler kh) {
          const VKLFeatureFlags featureFlags =
              kh.get_specialization_constant<samplerSpecId>();

          VKLHitIterator hitIterator = vklInitHitIterator(
              &hitContext, &rayOrigin, &rayDirection, &rayTRange, time,
              (void *)hitIteratorBuffer, featureFlags);

          for (;;) {
            VKLHit hit;
            int result = vklIterateHit(hitIterator, &hit, featureFlags);
            if (!result) {
              break;
            }
            hitBuffer[*numHits] = hit;

            *numHits = *numHits + 1;
            if (*numHits >= maxNumHits) break;
          }
        });
      })
      .wait();

  for (int i = 0; i < *numHits; ++i) {
    std::cout << "\t\tt " << hitBuffer[i].t << std::endl;
    std::cout << "\t\tsample " << hitBuffer[i].sample << std::endl;
    std::cout << "\t\tepsilon " << hitBuffer[i].epsilon << std::endl
              << std::endl;
  }

  sycl::free(hitIteratorBuffer, syclQueue);
  sycl::free(numHits, syclQueue);
  sycl::free(hitBuffer, syclQueue);

  vklRelease(hitContext);

  vklRelease(sampler);
}

int main() {
  auto IntelGPUDeviceSelector = [](const sycl::device &device) {
    using namespace sycl::info;
    const std::string deviceName = device.get_info<device::name>();
    bool match = device.is_gpu() &&
                 device.get_info<sycl::info::device::vendor_id>() == 0x8086 &&
                 device.get_backend() == sycl::backend::ext_oneapi_level_zero;
    return match ? 1 : -1;
  };

  sycl::queue syclQueue(IntelGPUDeviceSelector);

  sycl::context syclContext = syclQueue.get_context();

  std::cout << "Target SYCL device: "
            << syclQueue.get_device().get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  vklInit();

  VKLDevice device = vklNewDevice("gpu");
  vklDeviceSetVoidPtr(device, "syclContext", static_cast<void *>(&syclContext));
  vklCommitDevice(device);

  const int dimensions[] = {128, 128, 128};

  const int numVoxels = dimensions[0] * dimensions[1] * dimensions[2];

  const int numAttributes = 3;

  VKLVolume volume = vklNewVolume(device, "structuredRegular");
  vklSetVec3i(volume, "dimensions", dimensions[0], dimensions[1],
              dimensions[2]);
  vklSetVec3f(volume, "gridOrigin", 0, 0, 0);
  vklSetVec3f(volume, "gridSpacing", 1, 1, 1);

  std::vector<float> voxels(numVoxels);

  // volume attribute 0: x-grad
  for (int k = 0; k < dimensions[2]; k++)
    for (int j = 0; j < dimensions[1]; j++)
      for (int i = 0; i < dimensions[0]; i++)
        voxels[k * dimensions[0] * dimensions[1] + j * dimensions[2] + i] =
            (float)i;

  VKLData data0 = vklNewData(device, numVoxels, VKL_FLOAT, voxels.data());

  // volume attribute 1: y-grad
  for (int k = 0; k < dimensions[2]; k++)
    for (int j = 0; j < dimensions[1]; j++)
      for (int i = 0; i < dimensions[0]; i++)
        voxels[k * dimensions[0] * dimensions[1] + j * dimensions[2] + i] =
            (float)j;

  VKLData data1 = vklNewData(device, numVoxels, VKL_FLOAT, voxels.data());

  // volume attribute 2: z-grad
  for (int k = 0; k < dimensions[2]; k++)
    for (int j = 0; j < dimensions[1]; j++)
      for (int i = 0; i < dimensions[0]; i++)
        voxels[k * dimensions[0] * dimensions[1] + j * dimensions[2] + i] =
            (float)k;

  VKLData data2 = vklNewData(device, numVoxels, VKL_FLOAT, voxels.data());

  VKLData attributes[] = {data0, data1, data2};

  VKLData attributesData =
      vklNewData(device, numAttributes, VKL_DATA, attributes);

  vklRelease(data0);
  vklRelease(data1);
  vklRelease(data2);

  vklSetData(volume, "data", attributesData);
  vklRelease(attributesData);

  vklCommit(volume);

  demoGpuAPI(syclQueue, device, volume);

  vklRelease(volume);

  vklReleaseDevice(device);

  std::cout << "complete." << std::endl;

  return 0;
}
