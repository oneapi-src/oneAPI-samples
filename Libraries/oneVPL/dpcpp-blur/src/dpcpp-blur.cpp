//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) VPP application,
/// using 2.x API with internal memory management
///
/// @file

#include "util.h"

#ifdef BUILD_DPCPP
#include "CL/sycl.hpp"

#define BLUR_RADIUS 5
#define BLUR_SIZE (float)((BLUR_RADIUS << 1) + 1)

void BlurFrame(sycl::queue q, mfxFrameSurface1 *in_surface,
               mfxFrameSurface1 *blurred_surface);
#endif

#define OUTPUT_WIDTH 256
#define OUTPUT_HEIGHT 192
#define OUTPUT_FILE "out.raw"
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

void Usage(void) {
  printf("\n");
#ifdef BUILD_DPCPP
  printf(" ! Blur feature enabled by using DPCPP\n\n");
#else
  printf(" ! Blur feature disabled\n\n");
#endif
  printf("   Usage  :  dpcpp-blur\n");
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  dpcpp-blur -i in.i420 -w 128 -h 96 -sw\n");
  printf(
      "   To view:  ffplay -f rawvideo -pixel_format bgra -video_size %dx%d "
      "%s\n\n",
      OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FILE);
  printf("   Resize raw frames to %dx%d size in %s\n\n", OUTPUT_WIDTH,
         OUTPUT_HEIGHT, OUTPUT_FILE);

#ifdef BUILD_DPCPP
  printf(
      "   Blur VPP output by using DPCPP kernel (default kernel size is "
      "[%d]x[%d]) in %s\n",
      2 * BLUR_RADIUS + 1, 2 * BLUR_RADIUS + 1, OUTPUT_FILE);
#endif
  return;
}

#ifdef BUILD_DPCPP
// Few useful acronyms.
constexpr auto sycl_read = sycl::access::mode::read;
constexpr auto sycl_write = sycl::access::mode::write;

namespace dpc_common {
// this exception handler with catch async exceptions
static auto exception_handler = [](cl::sycl::exception_list exception_list) {
  for (std::exception_ptr const &e : exception_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};
};  // namespace dpc_common

// Select device on which to run kernel.
class MyDeviceSelector : public cl::sycl::device_selector {
 public:
  MyDeviceSelector() {}

  int operator()(const cl::sycl::device &device) const override {
    const std::string name = device.get_info<cl::sycl::info::device::name>();

    std::cout << "  Trying device: " << name << "..." << std::endl;
    std::cout << "  Vendor       : "
              << device.get_info<cl::sycl::info::device::vendor>() << std::endl
              << std::endl;

    if (device.is_gpu()) return 500;  // Higher merit for GPU
    if (device.is_cpu()) return 100;  // Select CPU if no GPU available

    return -1;
  }
};
#endif

int main(int argc, char *argv[]) {
  // Variables used for legacy and 2.x
  bool isDraining = false;
  bool isStillGoing = true;
  FILE *sink = NULL;
  FILE *source = NULL;
  mfxFrameSurface1 *vppInSurface = NULL;
  mfxFrameSurface1 *vppOutSurface = NULL;
  mfxSession session = NULL;
  mfxSyncPoint syncp = {};
  mfxU32 framenum = 0;
  mfxStatus sts = MFX_ERR_NONE;
  mfxStatus sts_r = MFX_ERR_NONE;
  Params cliParams = {};
  void *accelHandle = NULL;
  mfxVideoParam VPPParams = {};
  mfxU32 surface_size = 0;

  // variables used only in 2.x version
  mfxConfig cfg[3];
  mfxVariant cfgVal[3];
  mfxLoader loader = NULL;

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false) {
    Usage();
    return 1;  // return 1 as error code
  }

#ifdef BUILD_DPCPP
  printf("\n! DPCPP blur feature enabled\n\n");

  // Initialize DPC++
  MyDeviceSelector sel;

  mfxFrameSurface1 blurred_surface;
  std::vector<mfxU8> blur_data_out;
  // Create SYCL execution queue
  sycl::queue q(sel, dpc_common::exception_handler);

  // See what device was actually selected for this queue.
  // CPU is preferrable for this time.
  std::cout << "  Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl
            << std::endl;
#else
  printf("\n! DPCPP blur feature not enabled\n\n");
#endif

  source = fopen(cliParams.infileName, "rb");
  VERIFY(source, "Could not open input file");

  sink = fopen(OUTPUT_FILE, "wb");
  VERIFY(sink, "Could not create output file");

  // Initialize VPL session
  loader = MFXLoad();
  VERIFY(NULL != loader, "MFXLoad failed -- is implementation in path?");

  // Implementation used must be the type requested from command line
  cfg[0] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[0], "MFXCreateConfig failed")

  sts = MFXSetConfigFilterProperty(cfg[0], (mfxU8 *)"mfxImplDescription.Impl",
                                   cliParams.implValue);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed for Impl");

  // Implementation must provide VPP scaling
  cfg[1] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[1], "MFXCreateConfig failed")
  cfgVal[1].Type = MFX_VARIANT_TYPE_U32;
  cfgVal[1].Data.U32 = MFX_EXTBUFF_VPP_SCALING;
  sts = MFXSetConfigFilterProperty(
      cfg[1],
      (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
      cfgVal[1]);
  VERIFY(MFX_ERR_NONE == sts, "MFXSetConfigFilterProperty failed");

  // Implementation used must provide API version 2.2 or newer
  cfg[2] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[2], "MFXCreateConfig failed")
  cfgVal[2].Type = MFX_VARIANT_TYPE_U32;
  cfgVal[2].Data.U32 =
      VPLVERSION(MAJOR_API_VERSION_REQUIRED, MINOR_API_VERSION_REQUIRED);
  sts = MFXSetConfigFilterProperty(
      cfg[2], (mfxU8 *)"mfxImplDescription.ApiVersion.Version", cfgVal[2]);
  VERIFY(MFX_ERR_NONE == sts,
         "MFXSetConfigFilterProperty failed for API version");

  sts = MFXCreateSession(loader, 0, &session);
  VERIFY(MFX_ERR_NONE == sts,
         "Cannot create session -- no implementations meet selection criteria");

  // Print info about implementation loaded
  ShowImplementationInfo(loader, 0);

  // Convenience function to initialize available accelerator(s)
  accelHandle = InitAcceleratorHandle(session);

  // Initialize VPP parameters
  if (MFX_IMPL_SOFTWARE == cliParams.impl) {
    PrepareFrameInfo(&VPPParams.vpp.In, MFX_FOURCC_I420, cliParams.srcWidth,
                     cliParams.srcHeight);
    PrepareFrameInfo(&VPPParams.vpp.Out, MFX_FOURCC_BGRA, OUTPUT_WIDTH,
                     OUTPUT_HEIGHT);
  } else {
    PrepareFrameInfo(&VPPParams.vpp.In, MFX_FOURCC_NV12, cliParams.srcWidth,
                     cliParams.srcHeight);
    PrepareFrameInfo(&VPPParams.vpp.Out, MFX_FOURCC_BGRA, OUTPUT_WIDTH,
                     OUTPUT_HEIGHT);
  }

  VPPParams.IOPattern =
      MFX_IOPATTERN_IN_SYSTEM_MEMORY | MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

#ifdef BUILD_DPCPP
  surface_size = GetSurfaceSize(MFX_FOURCC_BGRA, OUTPUT_WIDTH, OUTPUT_HEIGHT);

  // Initialize surface for blurred frame
  blur_data_out.resize(surface_size);
  blurred_surface = {};
  blurred_surface.Info = VPPParams.vpp.Out;
  blurred_surface.Data.B = &blur_data_out[0];
  blurred_surface.Data.G = blurred_surface.Data.B + 1;
  blurred_surface.Data.R = blurred_surface.Data.G + 1;
  blurred_surface.Data.A = blurred_surface.Data.R + 1;
  blurred_surface.Data.Pitch = OUTPUT_WIDTH * 4;
#endif

  // Initialize VPP
  sts = MFXVideoVPP_Init(session, &VPPParams);
  VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

  printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  while (isStillGoing == true) {
    // Load a new frame if not draining
    if (isDraining == false) {
      sts = MFXMemory_GetSurfaceForVPPIn(session, &vppInSurface);
      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPIn");

      sts = ReadRawFrame_InternalMem(vppInSurface, source);
      if (sts == MFX_ERR_MORE_DATA)
        isDraining = true;
      else
        VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");

      sts = MFXMemory_GetSurfaceForVPPOut(session, &vppOutSurface);
      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPIn");
    }

    sts = MFXVideoVPP_RunFrameVPPAsync(
        session, (isDraining == true) ? NULL : vppInSurface, vppOutSurface,
        NULL, &syncp);

    if (!isDraining) {
      sts_r = vppInSurface->FrameInterface->Release(vppInSurface);
      VERIFY(MFX_ERR_NONE == sts_r, "mfxFrameSurfaceInterface->Release failed");
    }

    switch (sts) {
      case MFX_ERR_NONE:
        do {
          sts = vppOutSurface->FrameInterface->Synchronize(
              vppOutSurface, WAIT_100_MILLISECONDS);
          if (MFX_ERR_NONE == sts) {
#ifdef BUILD_DPCPP
            sts =
                vppOutSurface->FrameInterface->Map(vppOutSurface, MFX_MAP_READ);
            if (sts != MFX_ERR_NONE) {
              printf("mfxFrameSurfaceInterface->Map failed (%d)\n", sts);
              return sts;
            }

            BlurFrame(q, vppOutSurface, &blurred_surface);
            sts = WriteRawFrame(&blurred_surface, sink);
            if (sts != MFX_ERR_NONE) {
              printf("Error in WriteRawFrame\n");
              return sts;
            }

            sts = vppOutSurface->FrameInterface->Unmap(vppOutSurface);
            if (sts != MFX_ERR_NONE) {
              printf("mfxFrameSurfaceInterface->Unmap failed (%d)\n", sts);
              return sts;
            }

            sts = vppOutSurface->FrameInterface->Release(vppOutSurface);
            if (sts != MFX_ERR_NONE) {
              printf("mfxFrameSurfaceInterface->Release failed (%d)\n", sts);
              return sts;
            }
#else
            sts = WriteRawFrame_InternalMem(vppOutSurface, sink);
            VERIFY(MFX_ERR_NONE == sts, "Could not write vpp output");
#endif

            framenum++;
          }
        } while (sts == MFX_WRN_IN_EXECUTION);
        break;
      case MFX_ERR_MORE_DATA:
        // Need more input frames before VPP can produce an output
        if (isDraining) isStillGoing = false;
        break;
      case MFX_ERR_MORE_SURFACE:
        // Need more surfaces at output for additional output frames available.
        // This applies to external memory allocations and should not be
        // expected for a simple internal allocation case like this
        break;
      case MFX_ERR_DEVICE_LOST:
        // For non-CPU implementations,
        // Cleanup if device is lost
        break;
      case MFX_WRN_DEVICE_BUSY:
        // For non-CPU implementations,
        // Wait a few milliseconds then try again
        break;
      default:
        printf("unknown status %d\n", sts);
        isStillGoing = false;
        break;
    }
  }

end:
  printf("Processed %d frames\n", framenum);

  // Clean up resources - It is recommended to close components first, before
  // releasing allocated surfaces, since some surfaces may still be locked by
  // internal resources.
  if (source) fclose(source);

  if (sink) fclose(sink);

  MFXVideoVPP_Close(session);
  MFXClose(session);

  if (accelHandle) FreeAcceleratorHandle(accelHandle);

  if (loader) MFXUnload(loader);

  return 0;
}

#ifdef BUILD_DPCPP
// SYCL kernel scheduler
// Blur frame by using SYCL kernel
void BlurFrame(sycl::queue q, mfxFrameSurface1 *in_surface,
               mfxFrameSurface1 *blurred_surface) {
  int img_width, img_height;

  img_width = in_surface->Info.Width;
  img_height = in_surface->Info.Height;

  // Wrap mfx surfaces into SYCL image by using host ptr for zero copy of data
  sycl::image<2> image_buf_src(in_surface->Data.B,
                               sycl::image_channel_order::rgba,
                               sycl::image_channel_type::unsigned_int8,
                               sycl::range<2>(img_width, img_height));

  sycl::image<2> image_buf_dst(blurred_surface->Data.B,
                               sycl::image_channel_order::rgba,
                               sycl::image_channel_type::unsigned_int8,
                               sycl::range<2>(img_width, img_height));

  try {
    q.submit([&](cl::sycl::handler &cgh) {
      // Src image accessor
      sycl::accessor<cl::sycl::uint4, 2, sycl_read, sycl::access::target::image>
          accessor_src(image_buf_src, cgh);
      // Dst image accessor
      auto accessor_dst =
          image_buf_dst.get_access<cl::sycl::uint4, sycl_write>(cgh);
      cl::sycl::uint4 black = (cl::sycl::uint4)(0);
      // Parallel execution of the kerner for each pixel. Kernel
      // implemented as a lambda function.

      // Important: this is naive implementation of the blur kernel. For
      // further optimization it is better to use range_nd iterator and
      // apply moving average technique to reduce # of MAC operations per
      // pixel.
      cgh.parallel_for<class NaiveBlur_rgba>(
          sycl::range<2>(img_width, img_height), [=](sycl::item<2> item) {
            auto coords = cl::sycl::int2(item[0], item[1]);

            // Let's add horizontal black border
            if (item[0] <= BLUR_RADIUS ||
                item[0] >= img_width - 1 - BLUR_RADIUS) {
              accessor_dst.write(coords, black);
              return;
            }

            // Let's add vertical black border
            if (item[1] <= BLUR_RADIUS ||
                item[1] >= img_height - 1 - BLUR_RADIUS) {
              accessor_dst.write(coords, black);
              return;
            }

            cl::sycl::float4 tmp = (cl::sycl::float4)(0.f);
            cl::sycl::uint4 rgba;

            for (int i = item[0] - BLUR_RADIUS; i < item[0] + BLUR_RADIUS;
                 i++) {
              for (int j = item[1] - BLUR_RADIUS; j < item[1] + BLUR_RADIUS;
                   j++) {
                rgba = accessor_src.read(cl::sycl::int2(i, j));
                // Sum over the square mask
                tmp[0] += rgba.x();
                tmp[1] += rgba.y();
                tmp[2] += rgba.z();
                // Keep alpha channel from anchor pixel
                if (i == item[0] && j == item[1]) tmp[3] = rgba.w();
              }
            }
            // Compute average intensity
            tmp[0] /= BLUR_SIZE * BLUR_SIZE;
            tmp[1] /= BLUR_SIZE * BLUR_SIZE;
            tmp[2] /= BLUR_SIZE * BLUR_SIZE;

            // Convert and write blur pixel
            cl::sycl::uint4 tmp_u;
            tmp_u[0] = tmp[0];
            tmp_u[1] = tmp[1];
            tmp_u[2] = tmp[2];
            tmp_u[3] = tmp[3];

            accessor_dst.write(coords, tmp_u);
          });
    });

    // Since we are in blocking execution mode for this sample simplicity,
    // we need to wait for the execution completeness.
    q.wait_and_throw();
  } catch (std::exception e) {
    std::cout << "  SYCL exception caught: " << e.what() << std::endl;
    return;
  }
  return;
}
#endif
