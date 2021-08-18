//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) DPC++ interop application
/// using the core API subset.  For more information see:
/// https://software.intel.com/content/www/us/en/develop/articles/upgrading-from-msdk-to-onevpl.html
/// https://oneapi-src.github.io/oneAPI-spec/elements/oneVPL/source/index.html
///
/// @file

#include <CL/sycl.hpp>

#include "util.h"

#define OUTPUT_WIDTH 256
#define OUTPUT_HEIGHT 192
#define OUTPUT_FILE "out.raw"
#define BLUR_RADIUS 5
#define BLUR_SIZE (float)((BLUR_RADIUS << 1) + 1)
#define MAX_PLANES_NUMBER 4

#ifdef LIBVA_SUPPORT
  #define HAVE_VIDEO_MEMORY_INTEROP
#endif

#ifdef HAVE_VIDEO_MEMORY_INTEROP
  #include <unistd.h>

  #include <va/va.h>
  #include <va/va_drmcommon.h>

  #include <level_zero/ze_api.h>
  #include <CL/sycl/backend/level_zero.hpp>
#endif

// DPC++ kernel for image blurring
void BlurFrame(sycl::queue q, int width, int height, uint8_t *src_ptr,
               size_t src_stride, uint8_t *dst_ptr, size_t dst_stride) {
  try {
    q.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
       auto y = idx.get(0);
       auto x = idx.get(1);

       // Compute average intensity. Skip borders and set to black color
       float t0 = 0, t1 = 0, t2 = 0;

       if (x >= BLUR_RADIUS && x < (size_t)width - BLUR_RADIUS && y >= BLUR_RADIUS &&
           y < (size_t)height - BLUR_RADIUS) {
         for (size_t yy = y - BLUR_RADIUS; yy < y + BLUR_RADIUS; yy++) {
           for (size_t xx = x - BLUR_RADIUS; xx < x + BLUR_RADIUS; xx++) {
             t0 += src_ptr[yy * src_stride + 4 * xx];
             t1 += src_ptr[yy * src_stride + 4 * xx + 1];
             t2 += src_ptr[yy * src_stride + 4 * xx + 2];
           }
         }
         t0 /= BLUR_SIZE * BLUR_SIZE;
         t1 /= BLUR_SIZE * BLUR_SIZE;
         t2 /= BLUR_SIZE * BLUR_SIZE;
       }

       dst_ptr[y * dst_stride + 4 * x + 0] = t0;
       dst_ptr[y * dst_stride + 4 * x + 1] = t1;
       dst_ptr[y * dst_stride + 4 * x + 2] = t2;
       dst_ptr[y * dst_stride + 4 * x + 3] =
           src_ptr[y * src_stride + 4 * x + 3];
     }).wait_and_throw();
  } catch (std::exception e) {
    std::cout << "  SYCL exception caught: " << e.what() << std::endl;
    return;
  }
}

void Usage(char* app) {
  printf("\n");
  printf("   Usage  :  %s\n", app);
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  %s -i in.i420 -w 128 -h 96\n", app);
  printf(
      "   To view:  ffplay -f rawvideo -video_size %dx%d "
      "-pixel_format bgra %s\n\n",
      OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FILE);
  printf(" * Change color format to RGBA, resize raw frames to %dx%d size\n\n",
         OUTPUT_WIDTH, OUTPUT_HEIGHT);

  printf(
      "   Blur the VPP output by using DPCPP kernel (default kernel size is "
      "[%d]x[%d]) in %s\n",
      2 * BLUR_RADIUS + 1, 2 * BLUR_RADIUS + 1, OUTPUT_FILE);

  return;
}

int main(int argc, char **argv) {
  bool isDraining = false;
  bool isStillGoing = true;
  FILE *sink = NULL;
  FILE *source = NULL;
  mfxConfig cfg[1];
  mfxLoader loader = NULL;
  mfxSession session = NULL;
  mfxStatus sts = MFX_ERR_NONE;
  mfxSyncPoint syncp;
  mfxU32 blur_data_size = 0;
  mfxU32 framenum = 0;
  mfxU8 *blur_data = NULL;
  mfxVideoParam VPPParams = {};
  Params cliParams = {};
  size_t blur_pitch = 0;
  sycl::queue q;
#ifdef HAVE_VIDEO_MEMORY_INTEROP
  VADisplay va_dpy = NULL;
#endif
  mfxU32 desiredDevice = 0;
  // Let's make sure that we are searching oneVPL implementation
  // assotiated with sycl:queue we have already initialize.
  // cpu implementation has pci device id equals to 0.
  mfxU32 sessionIdx = 0;
  int desiredSessionIdx = -1;
  mfxImplDescription *implDesc;

  printf("Sample only: should not be used as benchmark.\n");

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false) {
    Usage(argv[0]);
    return 1;  // return 1 as error code
  }

#ifndef HAVE_VIDEO_MEMORY_INTEROP
  if (MFX_IMPL_TYPE_SOFTWARE != cliParams.implValue.Data.U32) {
    printf("Only software implementation is supported\n");
    return 1;
  }
#endif

  // Create SYCL execution queue
  q = (MFX_IMPL_SOFTWARE == cliParams.impl) ? sycl::queue(sycl::cpu_selector())
                                            : sycl::queue(sycl::gpu_selector());

  // Print device name selected for this queue.
  printf("Queue initialized on %s\n",
         q.get_device().get_info<sycl::info::device::name>().c_str());

#ifdef HAVE_VIDEO_MEMORY_INTEROP
  // Get Level-zero context and device from the SYCL backend
  ze_context_handle_t ze_context =
              q.get_context().get_native<sycl::backend::level_zero>();

  ze_device_handle_t ze_device =
              q.get_device().get_native<sycl::backend::level_zero>();
  if (q.get_device().get_info<sycl::info::device::device_type>() ==
                                      sycl::info::device_type::gpu) {
    ze_device_properties_t deviceProperties;
    zeDeviceGetProperties(ze_device, &deviceProperties);
    desiredDevice = (mfxU32)deviceProperties.deviceId;
  }
#endif

  source = fopen(cliParams.infileName, "rb");
  VERIFY(source, "Could not open input file");

  sink = fopen(OUTPUT_FILE, "wb");
  VERIFY(sink, "Could not create output file");

  // Allocate memory for blurred frame
  blur_data_size = GetSurfaceSize(MFX_FOURCC_BGRA, OUTPUT_WIDTH, OUTPUT_HEIGHT);
  blur_pitch = OUTPUT_WIDTH * 4;
  blur_data = sycl::malloc_shared<mfxU8>(blur_data_size, q);

  // Initialize oneVPL session
  loader = MFXLoad();
  VERIFY(NULL != loader, "MFXLoad failed -- is implementation in path?");

  // Implementation used must be the type requested from command line
  cfg[0] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[0], "MFXCreateConfig failed")

  sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl",
                                   cliParams.implValue);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

  // Match oneVPL implementation with sycl::q device
  while (MFX_ERR_NOT_FOUND != MFXEnumImplementations(loader, sessionIdx,
                  MFX_IMPLCAPS_IMPLDESCSTRUCTURE, (mfxHDL*)&implDesc)) {
    mfxU32 devID = (mfxU32)std::stoi(implDesc->Dev.DeviceID, nullptr, 16);
    if (devID == desiredDevice) {
      desiredSessionIdx = (int)sessionIdx;
      break;
    }
    MFXDispReleaseImplDescription(loader, implDesc);
    sessionIdx++;
  }

  VERIFY(sessionIdx >= 0,
         "Cannot create session -- no implementations meet selection criteria");

  sts = MFXCreateSession(loader, sessionIdx, &session);
  VERIFY(MFX_ERR_NONE == sts,
         "Cannot create session -- no implementations meet selection criteria");

  // Print info about implementation loaded
  ShowImplementationInfo(loader, 0);

  // Initialize VPP parameters:
  // 1. Describe VPP input
  PrepareFrameInfo(
      &VPPParams.vpp.In,
      (MFX_IMPL_SOFTWARE == cliParams.impl) ? MFX_FOURCC_I420 : MFX_FOURCC_NV12,
      cliParams.srcWidth, cliParams.srcHeight);
  // 2. Describe VPP output. We want VPP to resize the frame and cover color
  // space to RGBA.
  PrepareFrameInfo(&VPPParams.vpp.Out, MFX_FOURCC_BGRA, OUTPUT_WIDTH,
                   OUTPUT_HEIGHT);

  if (MFX_IMPL_SOFTWARE == cliParams.impl) {
    // Software based VPL implementation will allocate frames in system memory
    VPPParams.IOPattern =
        MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
  } else {
    // Hardware based VPL implementation will allocate frames in the device
    // memory. Later on we will try to reuse this memory in SYCL kernel
    // without any frame data copy (zero-copy).
#ifdef HAVE_VIDEO_MEMORY_INTEROP
    VPPParams.IOPattern =
        MFX_IOPATTERN_IN_VIDEO_MEMORY | MFX_IOPATTERN_OUT_VIDEO_MEMORY;
    // open VA display, set handle, and set allocator
    va_dpy = (VADisplay)InitAcceleratorHandle(session);
#endif
  }

  // Initialize VPP
  sts = MFXVideoVPP_Init(session, &VPPParams);
  VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

  // ===================================
  // Start processing the frames
  //

  printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  while (isStillGoing == true) {
    mfxFrameSurface1 *inSurface = nullptr, *outSurface = nullptr;

    if (isDraining == false) {
      // Allocate input surface for VPP
      sts = MFXMemory_GetSurfaceForVPPIn(session, &inSurface);
      VERIFY(MFX_ERR_NONE == sts, "Error in GetSurfaceForVPPIn");

      // Map surface to the system memory
      sts = inSurface->FrameInterface->Map(inSurface, MFX_MAP_WRITE);
      VERIFY(MFX_ERR_NONE == sts, "Error in Map for write");

      // Write input pixels to the surface
      sts = ReadRawFrame(inSurface,
                         source);  // Load frame from file into surface
      if (sts == MFX_ERR_MORE_DATA)
        // End of input file is reached. We need to switch to the drain mode
        // to let VPL output cached frames
        isDraining = true;
      else
        VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");
      
      // Unmap surface to the system memory
      sts = inSurface->FrameInterface->Unmap(inSurface);
      VERIFY(MFX_ERR_NONE == sts, "Error in Unmap");
    }

    // Allocate output surface for VPP
    sts = MFXMemory_GetSurfaceForVPPOut(session, &outSurface);
    VERIFY(MFX_ERR_NONE == sts, "Error in GetSurfaceForVPPOut");

    // Schedule resize and color space converion of input frame
    sts = MFXVideoVPP_RunFrameVPPAsync(
        session,
        (isDraining == true) ? NULL : inSurface,
        outSurface, NULL, &syncp);

    switch (sts) {
      case MFX_ERR_NONE: {
        // Wait for the frame processing complition.
        sts = MFXVideoCORE_SyncOperation(session, syncp,
                                         WAIT_100_MILLISECONDS * 1000);
        VERIFY(MFX_ERR_NONE == sts, "Error in SyncOperation");
#ifdef HAVE_VIDEO_MEMORY_INTEROP
        if ((VPPParams.IOPattern & MFX_IOPATTERN_IN_VIDEO_MEMORY) ==
            MFX_IOPATTERN_IN_VIDEO_MEMORY) {
          // In this code branch, we assume that VPL allocagted device memory
          // which we need to share with SYCL kernel through level-zero backend.
          mfxHDL handle = nullptr;
          mfxResourceType resource_type;

          // Retrieve native handle for the reviously allocated surface.
          // Ideally, we need to query the device but we have it already
          // in va_dpy variable
          sts = outSurface->FrameInterface->GetNativeHandle(outSurface,
                                                      &handle, &resource_type);
          VERIFY(MFX_ERR_NONE == sts, "Error in GetNativeHandle");

          // On Linux, we expect VASurfaceID as the native handle
          VERIFY(resource_type == MFX_RESOURCE_VA_SURFACE,
                           "Error: only MFX_RESOURCE_VA_SURFACE is supported");

          VASurfaceID va_surface_id = *(VASurfaceID *)handle;

          VADRMPRIMESurfaceDescriptor prime_desc = {};

          // Export DMA buffer file descriptor from libva library
          VAStatus va_sts = vaExportSurfaceHandle(
              va_dpy, va_surface_id, VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
              VA_EXPORT_SURFACE_READ_ONLY, &prime_desc);
          VERIFY(VA_STATUS_SUCCESS == va_sts, "error in vaExportHandle");

          // Check memory layout. At this moment we support only linear
          // memory layout.
          VERIFY( prime_desc.objects[0].drm_format_modifier == 0,
              "Error. Only linear memory layout is supported by SYCL kernel.");

          // Retrieve DMA  buf file descriptor to pass it to L0.
          int dma_fd = prime_desc.objects[0].fd;

          void *ptr = nullptr;
          ze_result_t ze_res;

          // Import DMA  buf file descriptor to L0 o convert it to USM.
          ze_device_mem_alloc_desc_t alloc_desc = {};
          alloc_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
          ze_external_memory_import_fd_t import_fd = {
              ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
              nullptr,  // pNext
              ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dma_fd};
          alloc_desc.pNext = &import_fd;
          ze_res =
              zeMemAllocDevice(ze_context, &alloc_desc,
                               prime_desc.objects[0].size, 0, ze_device, &ptr);

          VERIFY(ze_res == ZE_RESULT_SUCCESS,
             "Error: Failed to get USM pointer");

          // Execute SYCL kernel to blur the frame. Here we assume that
          // i/o frames are in device memory, so SYCL kernel is executed on GPU.
          BlurFrame(q, OUTPUT_WIDTH, OUTPUT_HEIGHT, (uint8_t *)ptr,
                    prime_desc.layers[0].pitch[0], blur_data, blur_pitch);

          // Store blured frame in the file.
          for (int r = 0; r < OUTPUT_HEIGHT; r++) {
            fwrite(blur_data + (r * blur_pitch), 1, blur_pitch, sink);
          }

          // unmap memory from L0
          ze_res = zeMemFree(ze_context, ptr);

          // close DMA buf file descriptor
          close(dma_fd);

        } else
#endif
        {
          // We don't have video memory interop supported or VPP output
          // already in the system memory.
          // In this case we need to map VPP'ed surface to the
          // system memory.
          sts = outSurface->FrameInterface->Map(outSurface, MFX_MAP_READ);
          VERIFY(MFX_ERR_NONE == sts, "Error in FrameInterface->Map");

          // Execute SYCL kernel to blur the frame. Here we assume that
          // i/o frames are in system memory, so SYCL kernel is executed on CPU.
          BlurFrame(q, OUTPUT_WIDTH, OUTPUT_HEIGHT, outSurface->Data.B,
                    outSurface->Data.Pitch, blur_data, blur_pitch);

          // Store blured frame in the file.
          for (int r = 0; r < OUTPUT_HEIGHT; r++) {
            fwrite(blur_data + (r * blur_pitch), 1, blur_pitch, sink);
          }

          // Unmap memory.
          sts = outSurface->FrameInterface->Unmap(outSurface);
          VERIFY(MFX_ERR_NONE == sts, "Error in FrameInterface->Unmap");
        }

        printf("Frame number: %d\r", ++framenum);
        fflush(stdout);
      } break;
      case MFX_ERR_MORE_DATA:
        // Need more input frames before VPP can produce an output
        if (isDraining) isStillGoing = false;
        break;
      case MFX_ERR_MORE_SURFACE:
        // The output frame is ready after synchronization.
        // Need more surfaces at output for additional output frames available.
        // This applies to external memory allocations and should not be
        // expected for a simple internal allocation case like this
        break;
      case MFX_ERR_DEVICE_LOST:
        // For non-CPU implementations,
        // Cleanup if device is lost
        break;
      case MFX_WRN_DEVICE_BUSY:
#ifdef HAVE_VIDEO_MEMORY_INTEROP
        // For non-CPU implementations,
        // Wait a few milliseconds then try again
        usleep(WAIT_100_MILLISECONDS * 1000);
#endif
        break;
      default:
        printf("unknown status %d\n", sts);
        isStillGoing = false;
        break;
    }
    // Release memory allocated for input surface
    sts = inSurface->FrameInterface->Release(inSurface);
    VERIFY(MFX_ERR_NONE == sts, "Error in FrameInterface->Release");

    // Release memory allocated for output surface
    sts = outSurface->FrameInterface->Release(outSurface);
    VERIFY(MFX_ERR_NONE == sts, "Error in FrameInterface->Release");
  }

end:
  printf("Processed %d frames\n", framenum);
  
  // Close i/o file
  if (source) fclose(source);
  if (sink) fclose(sink);

  // Release VPL resources
  if (session) {
    MFXVideoVPP_Close(session);
    MFXClose(session);
  }

  if (loader) MFXUnload(loader);

  // Release memory allocated for blured frame.
  sycl::free(blur_data, q);

  // Close the display.
#ifdef HAVE_VIDEO_MEMORY_INTEROP
  if (va_dpy) {
    FreeAcceleratorHandle(va_dpy);
  }
#endif

  return 0;
}
