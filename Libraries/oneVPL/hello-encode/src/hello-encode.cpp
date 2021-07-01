//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// A minimal oneAPI Video Processing Library (oneVPL) encode application,
/// using 2.x API with internal memory management
///
/// @file

#include "util.h"

#define TARGETKBPS 4000
#define FRAMERATE 30
#define OUTPUT_FILE "out.h265"
#define BITSTREAM_BUFFER_SIZE 2000000
#define MAJOR_API_VERSION_REQUIRED 2
#define MINOR_API_VERSION_REQUIRED 2

void Usage(void) {
  printf("\n");
  printf("   Usage  :  hello-encode\n");
  printf("     -hw        use hardware implementation\n");
  printf("     -sw        use software implementation\n");
  printf("     -i input file name (raw frames)\n");
  printf("     -w input width\n");
  printf("     -h input height\n\n");
  printf("   Example:  hello-encode -i in.i420 -w 128 -h 96\n");
  printf("   To view:  ffplay %s\n\n", OUTPUT_FILE);
  printf(" * Encode raw frames to HEVC/H265 elementary stream in %s\n\n",
         OUTPUT_FILE);
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
  mfxBitstream bitstream = {};
  mfxFrameSurface1 *encSurfaceIn = NULL;
  mfxSession session = NULL;
  mfxSyncPoint syncp = {};
  mfxU32 framenum = 0;
  mfxStatus sts = MFX_ERR_NONE;
  mfxStatus sts_r = MFX_ERR_NONE;
  Params cliParams = {};
  void *accelHandle = NULL;
  mfxVideoParam encodeParams = {};

  // variables used only in 2.x version
  mfxConfig cfg[3];
  mfxVariant cfgVal[3];
  mfxLoader loader = NULL;

  // Parse command line args to cliParams
  if (ParseArgsAndValidate(argc, argv, &cliParams, PARAMS_ENCODE) == false) {
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

  // Implementation must provide an HEVC encoder
  cfg[1] = MFXCreateConfig(loader);
  VERIFY(NULL != cfg[1], "MFXCreateConfig failed")
  cfgVal[1].Type = MFX_VARIANT_TYPE_U32;
  cfgVal[1].Data.U32 = MFX_CODEC_HEVC;
  sts = MFXSetConfigFilterProperty(
      cfg[1],
      (mfxU8 *)"mfxImplDescription.mfxEncoderDescription.encoder.CodecID",
      cfgVal[1]);
  VERIFY(MFX_ERR_NONE == sts,
         "MFXSetConfigFilterProperty failed for encoder CodecID");

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

  // Initialize encode parameters
  encodeParams.mfx.CodecId = MFX_CODEC_HEVC;
  encodeParams.mfx.TargetUsage = MFX_TARGETUSAGE_BALANCED;
  encodeParams.mfx.TargetKbps = TARGETKBPS;
  encodeParams.mfx.RateControlMethod = MFX_RATECONTROL_VBR;
  encodeParams.mfx.FrameInfo.FrameRateExtN = FRAMERATE;
  encodeParams.mfx.FrameInfo.FrameRateExtD = 1;
  if (MFX_IMPL_SOFTWARE == cliParams.impl) {
    encodeParams.mfx.FrameInfo.FourCC = MFX_FOURCC_I420;
  } else {
    encodeParams.mfx.FrameInfo.FourCC = MFX_FOURCC_NV12;
  }
  encodeParams.mfx.FrameInfo.ChromaFormat = MFX_CHROMAFORMAT_YUV420;
  encodeParams.mfx.FrameInfo.CropW = cliParams.srcWidth;
  encodeParams.mfx.FrameInfo.CropH = cliParams.srcHeight;
  encodeParams.mfx.FrameInfo.Width = ALIGN16(cliParams.srcWidth);
  encodeParams.mfx.FrameInfo.Height = ALIGN16(cliParams.srcHeight);

  encodeParams.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;

  // Initialize encoder
  sts = MFXVideoENCODE_Init(session, &encodeParams);
  VERIFY(MFX_ERR_NONE == sts, "Encode init failed");

  // Prepare output bitstream
  bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
  bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));

  printf("Encoding %s -> %s\n", cliParams.infileName, OUTPUT_FILE);

  printf("Input colorspace: ");
  switch (encodeParams.mfx.FrameInfo.FourCC) {
    case MFX_FOURCC_I420:  // CPU input
      printf("I420 (aka yuv420p)\n");
      break;
    case MFX_FOURCC_NV12:  // GPU input
      printf("NV12\n");
      break;
    default:
      printf("Unsupported color format\n");
      goto end;
      break;
  }

  while (isStillGoing == true) {
    // Load a new frame if not draining
    if (isDraining == false) {
      sts = MFXMemory_GetSurfaceForEncode(session, &encSurfaceIn);
      VERIFY(MFX_ERR_NONE == sts, "Could not get encode surface");

      sts = ReadRawFrame_InternalMem(encSurfaceIn, source);
      if (sts != MFX_ERR_NONE) isDraining = true;
    }

    sts = MFXVideoENCODE_EncodeFrameAsync(
        session, NULL, (isDraining == true) ? NULL : encSurfaceIn, &bitstream,
        &syncp);

    if (!isDraining) {
      sts_r = encSurfaceIn->FrameInterface->Release(encSurfaceIn);
      VERIFY(MFX_ERR_NONE == sts_r, "mfxFrameSurfaceInterface->Release failed");
    }
    switch (sts) {
      case MFX_ERR_NONE:
        // MFX_ERR_NONE and syncp indicate output is available
        if (syncp) {
          // Encode output is not available on CPU until sync operation
          // completes
          sts =
              MFXVideoCORE_SyncOperation(session, syncp, WAIT_100_MILLISECONDS);
          VERIFY(MFX_ERR_NONE == sts, "MFXVideoCORE_SyncOperation error");

          WriteEncodedStream(bitstream, sink);
          framenum++;
        }
        break;
      case MFX_ERR_NOT_ENOUGH_BUFFER:
        // This example deliberatly uses a large output buffer with immediate
        // write to disk for simplicity. Handle when frame size exceeds
        // available buffer here
        break;
      case MFX_ERR_MORE_DATA:
        // The function requires more data to generate any output
        if (isDraining == true) isStillGoing = false;
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
  printf("Encoded %d frames\n", framenum);

  // Clean up resources - It is recommended to close components first, before
  // releasing allocated surfaces, since some surfaces may still be locked by
  // internal resources.
  if (source) fclose(source);

  if (sink) fclose(sink);

  MFXVideoENCODE_Close(session);
  MFXClose(session);

  if (bitstream.Data) free(bitstream.Data);

  if (accelHandle) FreeAcceleratorHandle(accelHandle);

  if (loader) MFXUnload(loader);

  return 0;
}
