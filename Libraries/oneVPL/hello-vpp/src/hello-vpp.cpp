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

#define OUTPUT_WIDTH 640
#define OUTPUT_HEIGHT 480
#define OUTPUT_FILE "out.raw"
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 5
#define MAX_TIMEOUT_COUNT          10

void Usage(void) {
  printf("\n");
  printf("   Usage  :  hello-vpp\n");
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (sw=I420 raw frames, hw=NV12)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  hello-vpp -i in.i420 -w 128 -h 96 -sw\n");
  printf(
      "   To view:  ffplay -f rawvideo -pixel_format bgra -video_size %dx%d "
      "%s\n\n",
      OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_FILE);
  printf(" * Resize raw frames to %dx%d size in %s\n\n", OUTPUT_WIDTH,
         OUTPUT_HEIGHT, OUTPUT_FILE);
  printf(
      "   CPU native color format is I420/yuv420p.  GPU native color format is "
      "NV12\n");
  return;
}

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

  // variables used only in 2.x version
  mfxConfig cfg[3];
  mfxVariant cfgVal[3];
  mfxLoader loader = NULL;
  mfxU8 timeout_count;

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_VPP) == false) {
    Usage();
    return 1;  // return 1 as error code
  }

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

  // Implementation must provide equal to or higher API version than
  // MAJOR_API_VERSION_REQUIRED.MINOR_API_VERSION_REQUIRED
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

  // Initialize VPP
  sts = MFXVideoVPP_Init(session, &VPPParams);
  VERIFY(MFX_ERR_NONE == sts, "Could not initialize VPP");

  printf("Processing %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  while (isStillGoing == true) {
    // Load a new frame if not draining
    if (isDraining == false) {
      timeout_count = 0;
      do {
        sts = MFXMemory_GetSurfaceForVPPIn(session, &vppInSurface);
        // From API version 2.5,
        // When the internal memory model is used,
        // MFX_WRN_ALLOC_TIMEOUT_EXPIRED is returned when all the surfaces are currently in use
        // and timeout set by mfxExtAllocationHints for allocation of new surfaces through functions
        // GetSurfaceForXXX/RunFrameAsync/DecodeFrameAsync expired.
        // Repeat the call in a few milliseconds.
        // For more information, please check oneVPL API documentation.
        if (sts == MFX_WRN_ALLOC_TIMEOUT_EXPIRED) {
            if (timeout_count > MAX_TIMEOUT_COUNT) {
                sts = MFX_ERR_DEVICE_FAILED;
                break;
            }
            else {
                timeout_count++;
                sleep(WAIT_5_MILLISECONDS);
                continue;
            }
        }
        else
            break;
      } while (1);

      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPIn");

      sts = ReadRawFrame_InternalMem(vppInSurface, source);
      if (sts == MFX_ERR_MORE_DATA)
        isDraining = true;
      else
        VERIFY(MFX_ERR_NONE == sts, "Unknown error reading input");

      timeout_count = 0;
      do {
        sts = MFXMemory_GetSurfaceForVPPOut(session, &vppOutSurface);
        if (sts == MFX_WRN_ALLOC_TIMEOUT_EXPIRED) {
            if (timeout_count > MAX_TIMEOUT_COUNT) {
                sts = MFX_ERR_DEVICE_FAILED;
                break;
            }
            else {
                timeout_count++;
                sleep(WAIT_5_MILLISECONDS);
                continue;
            }
        }
        else
            break;
      } while (1);

      VERIFY(MFX_ERR_NONE == sts,
             "Unknown error in MFXMemory_GetSurfaceForVPPOut");
    }

    timeout_count = 0;
    do {
      sts = MFXVideoVPP_RunFrameVPPAsync(
          session, (isDraining == true) ? NULL : vppInSurface, vppOutSurface,
          NULL, &syncp);
      if (sts == MFX_WRN_ALLOC_TIMEOUT_EXPIRED) {
          if (timeout_count > MAX_TIMEOUT_COUNT) {
              sts = MFX_ERR_DEVICE_FAILED;
              break;
          }
          else {
              timeout_count++;
              sleep(WAIT_5_MILLISECONDS);
              continue;
          }
      }
      else
          break;
    } while (1);

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
            sts = WriteRawFrame_InternalMem(vppOutSurface, sink);
            VERIFY(MFX_ERR_NONE == sts, "Could not write vpp output");

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
