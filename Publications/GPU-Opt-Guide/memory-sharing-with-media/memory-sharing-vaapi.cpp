//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// SYCL
#include <CL/sycl.hpp>

// SYCL oneAPI extension
#include <sycl/ext/oneapi/backend/level_zero.hpp>

// Level-zero
#include <level_zero/ze_api.h>

// VA-API
#include <va/va_drm.h>
#include <va/va_drmcommon.h>

#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#define OUTPUT_FILE "output.bgra"

#define VAAPI_DEVICE "/dev/dri/renderD128"

#define FRAME_WIDTH 320
#define FRAME_HEIGHT 240

#define RECT_WIDTH 160
#define RECT_HEIGHT 160
#define RECT_Y (FRAME_HEIGHT - RECT_HEIGHT) / 2

#define NUM_FRAMES (FRAME_WIDTH - RECT_WIDTH)

#define VA_FORMAT VA_FOURCC_BGRA
#define RED 0xffff0000
#define GREEN 0xff00ff00
#define BLUE 0xff0000ff

#define CHECK_STS(_FUNC)                                                       \
  {                                                                            \
    auto _sts = _FUNC;                                                         \
    if (_sts != 0) {                                                           \
      printf("Error %d calling " #_FUNC, (int)_sts);                           \
      return -1;                                                               \
    }                                                                          \
  }

VASurfaceID alloc_va_surface(VADisplay va_display, int width, int height) {
  VASurfaceID va_surface;
  VASurfaceAttrib surface_attrib{};
  surface_attrib.type = VASurfaceAttribPixelFormat;
  surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
  surface_attrib.value.type = VAGenericValueTypeInteger;
  surface_attrib.value.value.i = VA_FORMAT;
  vaCreateSurfaces(va_display, VA_RT_FORMAT_RGB32, width, height, &va_surface,
                   1, &surface_attrib, 1);
  return va_surface;
}

int main() {
  // Create SYCL queue on GPU device and Level-zero backend, and query
  // Level-zero context and device
  sycl::queue sycl_queue{sycl::ext::oneapi::filter_selector(
      "level_zero")}; // { sycl::gpu_selector() }
  auto ze_context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      sycl_queue.get_context());
  auto ze_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
      sycl_queue.get_device());

  // Create VA-API context (VADisplay)
  VADisplay va_display = vaGetDisplayDRM(open(VAAPI_DEVICE, O_RDWR));
  if (!va_display) {
    printf("Error creating VADisplay on device %s\n", VAAPI_DEVICE);
    return -1;
  }
  int major = 0, minor = 0;
  CHECK_STS(vaInitialize(va_display, &major, &minor));

  // Create VA-API surfaces
  VASurfaceID surfaces[NUM_FRAMES];
  for (int i = 0; i < NUM_FRAMES; i++) {
    surfaces[i] = alloc_va_surface(va_display, FRAME_WIDTH, FRAME_HEIGHT);
  }

  // Convert each VA-API surface into USM device pointer (zero-copy buffer
  // sharing between VA-API and Level-zero)
  void *device_ptr[NUM_FRAMES];
  size_t stride;
  for (int i = 0; i < NUM_FRAMES; i++) {
    // Export DMA-FD from VASurface
    VADRMPRIMESurfaceDescriptor prime_desc{};
    CHECK_STS(vaExportSurfaceHandle(va_display, surfaces[i],
                                    VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                                    VA_EXPORT_SURFACE_READ_WRITE, &prime_desc));
    auto dma_fd = prime_desc.objects->fd;
    auto dma_size = prime_desc.objects->size;
    stride = prime_desc.layers[0].pitch[0] / sizeof(uint32_t);

    // Import DMA-FD into Level-zero device pointer
    ze_external_memory_import_fd_t import_fd = {
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
        nullptr, // pNext
        ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dma_fd};
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = &import_fd;
    CHECK_STS(zeMemAllocDevice(ze_context, &alloc_desc, dma_size, 1, ze_device,
                               &device_ptr[i]));

    // Close DMA-FD
    close(dma_fd);
  }

  // Create VA-API surface with size 1x1 and write GREEN pixel
  VASurfaceID surface1x1 = alloc_va_surface(va_display, 1, 1);
  VAImage va_image;
  void *data = nullptr;
  CHECK_STS(vaDeriveImage(va_display, surface1x1, &va_image));
  CHECK_STS(vaMapBuffer(va_display, va_image.buf, &data));
  *(uint32_t *)data = GREEN;
  CHECK_STS(vaUnmapBuffer(va_display, va_image.buf));
  CHECK_STS(vaDestroyImage(va_display, va_image.image_id));

  // VA-API call to fill background with BLUE color and upscale 1x1 surface into
  // moving GREEN rectangle
  VAConfigID va_config_id;
  VAContextID va_context_id;
  CHECK_STS(vaCreateConfig(va_display, VAProfileNone, VAEntrypointVideoProc,
                           nullptr, 0, &va_config_id));
  CHECK_STS(vaCreateContext(va_display, va_config_id, 0, 0, VA_PROGRESSIVE,
                            nullptr, 0, &va_context_id));
  for (int i = 0; i < NUM_FRAMES; i++) {
    VAProcPipelineParameterBuffer param{};
    param.output_background_color = BLUE;
    param.surface = surface1x1;
    VARectangle output_region = {int16_t(i), RECT_Y, RECT_WIDTH, RECT_HEIGHT};
    param.output_region = &output_region;
    VABufferID param_buf;
    CHECK_STS(vaCreateBuffer(va_display, va_context_id,
                             VAProcPipelineParameterBufferType, sizeof(param),
                             1, &param, &param_buf));
    CHECK_STS(vaBeginPicture(va_display, va_context_id, surfaces[i]));
    CHECK_STS(vaRenderPicture(va_display, va_context_id, &param_buf, 1));
    CHECK_STS(vaEndPicture(va_display, va_context_id));
    CHECK_STS(vaDestroyBuffer(va_display, param_buf));
  }

#if 0
    // Synchronization is optional on Linux OS as i915 KMD driver synchronizes
    // write/read commands submitted from Intel media and compute drivers
    for (int i = 0; i < NUM_FRAMES; i++) {
        CHECK_STS(vaSyncSurface(va_display, surfaces[i]));
    }
#endif

  // Submit SYCL kernels to write RED sub-rectangle inside GREEN rectangle
  std::vector<sycl::event> sycl_events(NUM_FRAMES);
  for (int i = 0; i < NUM_FRAMES; i++) {
    uint32_t *ptr = (uint32_t *)device_ptr[i] +
                    (RECT_Y + RECT_HEIGHT / 4) * stride + (i + RECT_WIDTH / 4);
    sycl_events[i] = sycl_queue.parallel_for(
        sycl::range<2>(RECT_HEIGHT / 2, RECT_WIDTH / 2), [=](sycl::id<2> idx) {
          auto y = idx.get(0);
          auto x = idx.get(1);
          ptr[y * stride + x] = RED;
        });
  }

  // Synchronize all SYCL kernels
  sycl::event::wait(sycl_events);

  // Map VA-API surface to system memory and write to file
  FILE *file = fopen(OUTPUT_FILE, "wb");
  if (!file) {
    printf("Error creating file %s\n", OUTPUT_FILE);
    return -1;
  }
  for (int i = 0; i < NUM_FRAMES; i++) {
    CHECK_STS(vaDeriveImage(va_display, surfaces[i], &va_image));
    CHECK_STS(vaMapBuffer(va_display, va_image.buf, &data));
    fwrite(data, 1, FRAME_HEIGHT * FRAME_WIDTH * 4, file);
    CHECK_STS(vaUnmapBuffer(va_display, va_image.buf));
    CHECK_STS(vaDestroyImage(va_display, va_image.image_id));
  }
  fclose(file);
  printf("Created file %s\n", OUTPUT_FILE);

  // Free device pointers and VA-API surfaces
  for (int i = 0; i < NUM_FRAMES; i++)
    zeMemFree(ze_context, device_ptr[i]);
  vaDestroySurfaces(va_display, surfaces, NUM_FRAMES);

  return 0;
}
