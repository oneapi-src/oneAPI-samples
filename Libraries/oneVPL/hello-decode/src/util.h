//==============================================================================
// Copyright Intel Corporation
//
// SPDX-License-Identifier: MIT
//==============================================================================

///
/// Utility library header file for sample code
///
/// @file

#ifndef EXAMPLES_UTIL_H_
#define EXAMPLES_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_MEDIASDK1
#include "mfxvideo.h"
enum {
  MFX_FOURCC_I420 = MFX_FOURCC_IYUV /*!< Alias for the IYUV color format. */
};
#else
#include "vpl/mfxjpeg.h"
#include "vpl/mfxvideo.h"
#endif

#if (MFX_VERSION >= 2000)
#include "vpl/mfxdispatcher.h"
#endif

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef LIBVA_SUPPORT
#include "va/va.h"
#include "va/va_drm.h"
#endif

#define WAIT_100_MILLISECONDS 100
#define MAX_PATH 260
#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160
#define IS_ARG_EQ(a, b) (!strcmp((a), (b)))

#define VERIFY(x, y)   \
  if (!(x)) {          \
    printf("%s\n", y); \
    goto end;          \
  }

#define ALIGN16(value) (((value + 15) >> 4) << 4)
#define ALIGN32(X) (((mfxU32)((X) + 31)) & (~(mfxU32)31))
#define VPLVERSION(major, minor) (major << 16 | minor)

enum ExampleParams { PARAM_IMPL = 0, PARAM_INFILE, PARAM_INRES, PARAM_COUNT };
enum ParamGroup {
  PARAMS_CREATESESSION = 0,
  PARAMS_DECODE,
  PARAMS_ENCODE,
  PARAMS_VPP,
  PARAMS_TRANSCODE
};

typedef struct _Params {
  mfxIMPL impl;
#if (MFX_VERSION >= 2000)
  mfxVariant implValue;
#endif

  char *infileName;
  char *inmodelName;

  mfxU16 srcWidth;
  mfxU16 srcHeight;
} Params;

char *ValidateFileName(char *in) {
  if (in) {
    if (strnlen(in, MAX_PATH) > MAX_PATH) return NULL;
  }

  return in;
}

bool ValidateSize(char *in, mfxU16 *vsize, mfxU32 vmax) {
  if (in) {
    *vsize = static_cast<mfxU16>(strtol(in, NULL, 10));
    if (*vsize <= vmax) return true;
  }

  *vsize = 0;
  return false;
}

bool ParseArgsAndValidate(int argc, char *argv[], Params *params,
                          ParamGroup group) {
  int idx;
  char *s;

  // init all params to 0
  *params = {};
  params->impl = MFX_IMPL_SOFTWARE;
#if (MFX_VERSION >= 2000)
  params->implValue.Type = MFX_VARIANT_TYPE_U32;
  params->implValue.Data.U32 = MFX_IMPL_TYPE_SOFTWARE;
#endif

  for (idx = 1; idx < argc;) {
    // all switches must start with '-'
    if (argv[idx][0] != '-') {
      printf("ERROR - invalid argument: %s\n", argv[idx]);
      return false;
    }

    // switch string, starting after the '-'
    s = &argv[idx][1];
    idx++;

    // search for match
    if (IS_ARG_EQ(s, "i")) {
      params->infileName = ValidateFileName(argv[idx++]);
      if (!params->infileName) {
        return false;
      }
    } else if (IS_ARG_EQ(s, "m")) {
      params->inmodelName = ValidateFileName(argv[idx++]);
      if (!params->inmodelName) {
        return false;
      }
    } else if (IS_ARG_EQ(s, "w")) {
      if (!ValidateSize(argv[idx++], &params->srcWidth, MAX_WIDTH))
        return false;
    } else if (IS_ARG_EQ(s, "h")) {
      if (!ValidateSize(argv[idx++], &params->srcHeight, MAX_HEIGHT))
        return false;
    } else if (IS_ARG_EQ(s, "hw")) {
      params->impl = MFX_IMPL_HARDWARE;
#if (MFX_VERSION >= 2000)
      params->implValue.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
#endif
    } else if (IS_ARG_EQ(s, "sw")) {
      params->impl = MFX_IMPL_SOFTWARE;
#if (MFX_VERSION >= 2000)
      params->implValue.Data.U32 = MFX_IMPL_TYPE_SOFTWARE;
#endif
    }
  }

  // input file required by all except createsession
  if ((group != PARAMS_CREATESESSION) && (!params->infileName)) {
    printf("ERROR - input file name (-i) is required\n");
    return false;
  }

  // VPP and encode samples require an input resolution
  if ((PARAMS_VPP == group) || (PARAMS_ENCODE == group)) {
    if ((!params->srcWidth) || (!params->srcHeight)) {
      printf("ERROR - source width/height required\n");
      return false;
    }
  }

  return true;
}

void *InitAcceleratorHandle(mfxSession session) {
  mfxIMPL impl;
  mfxStatus sts = MFXQueryIMPL(session, &impl);
  if (sts != MFX_ERR_NONE) return NULL;

#ifdef LIBVA_SUPPORT
  if ((impl & MFX_IMPL_VIA_VAAPI) == MFX_IMPL_VIA_VAAPI) {
    VADisplay va_dpy = NULL;
    int fd;
    // initialize VAAPI context and set session handle (req in Linux)
    fd = open("/dev/dri/renderD128", O_RDWR);
    if (fd >= 0) {
      va_dpy = vaGetDisplayDRM(fd);
      if (va_dpy) {
        int major_version = 0, minor_version = 0;
        if (VA_STATUS_SUCCESS ==
            vaInitialize(va_dpy, &major_version, &minor_version)) {
          MFXVideoCORE_SetHandle(
              session, static_cast<mfxHandleType>(MFX_HANDLE_VA_DISPLAY),
              va_dpy);
        }
      }
    }
    return va_dpy;
  }
#endif

  return NULL;
}

void FreeAcceleratorHandle(void *accelHandle) {
#ifdef LIBVA_SUPPORT
  vaTerminate((VADisplay)accelHandle);
#endif
}

// Shows implementation info for Media SDK or oneVPL
mfxVersion ShowImplInfo(mfxSession session) {
  mfxIMPL impl;
  mfxVersion version = {0, 1};

  mfxStatus sts = MFXQueryIMPL(session, &impl);
  if (sts != MFX_ERR_NONE) return version;

  sts = MFXQueryVersion(session, &version);
  if (sts != MFX_ERR_NONE) return version;

  printf("Session loaded: ApiVersion = %d.%d \timpl= ", version.Major,
         version.Minor);

  switch (impl) {
    case MFX_IMPL_SOFTWARE:
      puts("Software");
      break;
    case MFX_IMPL_HARDWARE | MFX_IMPL_VIA_VAAPI:
      puts("Hardware:VAAPI");
      break;
    case MFX_IMPL_HARDWARE | MFX_IMPL_VIA_D3D11:
      puts("Hardware:D3D11");
      break;
    case MFX_IMPL_HARDWARE | MFX_IMPL_VIA_D3D9:
      puts("Hardware:D3D9");
      break;
    default:
      puts("Unknown");
      break;
  }

  return version;
}

// Shows implementation info with oneVPL
void ShowImplementationInfo(mfxLoader loader, mfxU32 implnum) {
  mfxImplDescription *idesc = nullptr;
  mfxStatus sts;
  // Loads info about implementation at specified list location
  sts = MFXEnumImplementations(loader, implnum, MFX_IMPLCAPS_IMPLDESCSTRUCTURE,
                               (mfxHDL *)&idesc);
  if (!idesc || (sts != MFX_ERR_NONE)) return;

  printf("Implementation details:\n");
  printf("  ApiVersion:           %hu.%hu  \n", idesc->ApiVersion.Major,
         idesc->ApiVersion.Minor);
  printf("  Implementation type:  %s\n",
         (idesc->Impl == MFX_IMPL_TYPE_SOFTWARE) ? "SW" : "HW");
  printf("  AccelerationMode via: ");
  switch (idesc->AccelerationMode) {
    case MFX_ACCEL_MODE_NA:
      printf("NA \n");
      break;
    case MFX_ACCEL_MODE_VIA_D3D9:
      printf("D3D9\n");
      break;
    case MFX_ACCEL_MODE_VIA_D3D11:
      printf("D3D11\n");
      break;
    case MFX_ACCEL_MODE_VIA_VAAPI:
      printf("VAAPI\n");
      break;
    case MFX_ACCEL_MODE_VIA_VAAPI_DRM_MODESET:
      printf("VAAPI_DRM_MODESET\n");
      break;
    case MFX_ACCEL_MODE_VIA_VAAPI_GLX:
      printf("VAAPI_GLX\n");
      break;
    case MFX_ACCEL_MODE_VIA_VAAPI_X11:
      printf("VAAPI_X11\n");
      break;
    case MFX_ACCEL_MODE_VIA_VAAPI_WAYLAND:
      printf("VAAPI_WAYLAND\n");
      break;
    case MFX_ACCEL_MODE_VIA_HDDLUNITE:
      printf("HDDLUNITE\n");
      break;
    default:
      printf("unknown\n");
      break;
  }
  MFXDispReleaseImplDescription(loader, idesc);

#if (MFX_VERSION >= 2004)
  // Show implementation path, added in 2.4 API
  mfxHDL implPath = nullptr;
  sts =
      MFXEnumImplementations(loader, implnum, MFX_IMPLCAPS_IMPLPATH, &implPath);
  if (!implPath || (sts != MFX_ERR_NONE)) return;

  printf("  Path: %s\n\n", reinterpret_cast<mfxChar *>(implPath));
  MFXDispReleaseImplDescription(loader, implPath);
#endif
}

void PrepareFrameInfo(mfxFrameInfo *fi, mfxU32 format, mfxU16 w, mfxU16 h) {
  // Video processing input data format
  fi->FourCC = format;
  fi->ChromaFormat = MFX_CHROMAFORMAT_YUV420;
  fi->CropX = 0;
  fi->CropY = 0;
  fi->CropW = w;
  fi->CropH = h;
  fi->PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
  fi->FrameRateExtN = 30;
  fi->FrameRateExtD = 1;
  // width must be a multiple of 16
  // height must be a multiple of 16 in case of frame picture and a multiple of
  // 32 in case of field picture
  fi->Width = ALIGN16(fi->CropW);
  fi->Height = (MFX_PICSTRUCT_PROGRESSIVE == fi->PicStruct)
                   ? ALIGN16(fi->CropH)
                   : ALIGN32(fi->CropH);
}

mfxU32 GetSurfaceSize(mfxU32 FourCC, mfxU32 width, mfxU32 height) {
  mfxU32 nbytes = 0;

  switch (FourCC) {
    case MFX_FOURCC_I420:
    case MFX_FOURCC_NV12:
      nbytes = width * height + (width >> 1) * (height >> 1) +
               (width >> 1) * (height >> 1);
      break;
    case MFX_FOURCC_I010:
    case MFX_FOURCC_P010:
      nbytes = width * height + (width >> 1) * (height >> 1) +
               (width >> 1) * (height >> 1);
      nbytes *= 2;
      break;
    case MFX_FOURCC_RGB4:
      nbytes = width * height * 4;
      break;
    default:
      break;
  }

  return nbytes;
}

int GetFreeSurfaceIndex(mfxFrameSurface1 *SurfacesPool, mfxU16 nPoolSize) {
  for (mfxU16 i = 0; i < nPoolSize; i++) {
    if (0 == SurfacesPool[i].Data.Locked) return i;
  }
  return MFX_ERR_NOT_FOUND;
}

mfxStatus AllocateExternalSystemMemorySurfacePool(mfxU8 **buf,
                                                  mfxFrameSurface1 *surfpool,
                                                  mfxFrameInfo frame_info,
                                                  mfxU16 surfnum) {
  // initialize surface pool (I420, RGB4 format)
  mfxU32 surfaceSize =
      GetSurfaceSize(frame_info.FourCC, frame_info.Width, frame_info.Height);
  if (!surfaceSize) return MFX_ERR_MEMORY_ALLOC;

  size_t framePoolBufSize = static_cast<size_t>(surfaceSize) * surfnum;
  *buf = reinterpret_cast<mfxU8 *>(calloc(framePoolBufSize, 1));

  mfxU16 surfW;
  mfxU16 surfH = frame_info.Height;

  if (frame_info.FourCC == MFX_FOURCC_RGB4) {
    surfW = frame_info.Width * 4;

    for (mfxU32 i = 0; i < surfnum; i++) {
      surfpool[i] = {0};
      surfpool[i].Info = frame_info;
      size_t buf_offset = static_cast<size_t>(i) * surfaceSize;
      surfpool[i].Data.B = *buf + buf_offset;
      surfpool[i].Data.G = surfpool[i].Data.B + 1;
      surfpool[i].Data.R = surfpool[i].Data.B + 2;
      surfpool[i].Data.A = surfpool[i].Data.B + 3;
      surfpool[i].Data.Pitch = surfW;
    }
  } else {
    surfW = (frame_info.FourCC == MFX_FOURCC_P010) ? frame_info.Width * 2
                                                   : frame_info.Width;

    for (mfxU32 i = 0; i < surfnum; i++) {
      surfpool[i] = {0};
      surfpool[i].Info = frame_info;
      size_t buf_offset = static_cast<size_t>(i) * surfaceSize;
      surfpool[i].Data.Y = *buf + buf_offset;
      surfpool[i].Data.U = *buf + buf_offset + (surfW * surfH);
      surfpool[i].Data.V = surfpool[i].Data.U + ((surfW / 2) * (surfH / 2));
      surfpool[i].Data.Pitch = surfW;
    }
  }

  return MFX_ERR_NONE;
}

void FreeExternalSystemMemorySurfacePool(mfxU8 *dec_buf,
                                         mfxFrameSurface1 *surfpool) {
  if (dec_buf) free(dec_buf);

  if (surfpool) free(surfpool);
}

// Read encoded stream from file
mfxStatus ReadEncodedStream(mfxBitstream &bs, FILE *f) {
  mfxU8 *p0 = bs.Data;
  mfxU8 *p1 = bs.Data + bs.DataOffset;
  if (bs.DataOffset > bs.MaxLength - 1) {
    return MFX_ERR_NOT_ENOUGH_BUFFER;
  }
  if (bs.DataLength + bs.DataOffset > bs.MaxLength) {
    return MFX_ERR_NOT_ENOUGH_BUFFER;
  }
  for (mfxU32 i = 0; i < bs.DataLength; i++) {
    *(p0++) = *(p1++);
  }
  bs.DataOffset = 0;
  bs.DataLength += (mfxU32)fread(bs.Data + bs.DataLength, 1,
                                 bs.MaxLength - bs.DataLength, f);
  if (bs.DataLength == 0) return MFX_ERR_MORE_DATA;

  return MFX_ERR_NONE;
}

// Write encoded stream to file
void WriteEncodedStream(mfxBitstream &bs, FILE *f) {
  fwrite(bs.Data + bs.DataOffset, 1, bs.DataLength, f);
  bs.DataLength = 0;
  return;
}

// Load raw I420 frames to mfxFrameSurface
mfxStatus ReadRawFrame(mfxFrameSurface1 *surface, FILE *f) {
  mfxU16 w, h, i, pitch;
  size_t bytes_read;
  mfxU8 *ptr;
  mfxFrameInfo *info = &surface->Info;
  mfxFrameData *data = &surface->Data;

  w = info->Width;
  h = info->Height;

  switch (info->FourCC) {
    case MFX_FOURCC_I420:
      // read luminance plane (Y)
      pitch = data->Pitch;
      ptr = data->Y;
      for (i = 0; i < h; i++) {
        bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
        if (w != bytes_read) return MFX_ERR_MORE_DATA;
      }

      // read chrominance (U, V)
      pitch /= 2;
      h /= 2;
      w /= 2;
      ptr = data->U;
      for (i = 0; i < h; i++) {
        bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
        if (w != bytes_read) return MFX_ERR_MORE_DATA;
      }

      ptr = data->V;
      for (i = 0; i < h; i++) {
        bytes_read = (mfxU32)fread(ptr + i * pitch, 1, w, f);
        if (w != bytes_read) return MFX_ERR_MORE_DATA;
      }
      break;
    case MFX_FOURCC_NV12:
      // Y
      pitch = data->Pitch;
      for (i = 0; i < h; i++) {
        bytes_read = fread(data->Y + i * pitch, 1, w, f);
        if (w != bytes_read) return MFX_ERR_MORE_DATA;
      }
      // UV
      h /= 2;
      for (i = 0; i < h; i++) {
        bytes_read = fread(data->UV + i * pitch, 1, w, f);
        if (w != bytes_read) return MFX_ERR_MORE_DATA;
      }
      break;
    case MFX_FOURCC_RGB4:
      // Y
      pitch = data->Pitch;
      for (i = 0; i < h; i++) {
        bytes_read = fread(data->B + i * pitch, 1, pitch, f);
        if (pitch != bytes_read) return MFX_ERR_MORE_DATA;
      }
      break;
    default:
      printf("Unsupported FourCC code, skip LoadRawFrame\n");
      break;
  }

  return MFX_ERR_NONE;
}

#if (MFX_VERSION >= 2000)
mfxStatus ReadRawFrame_InternalMem(mfxFrameSurface1 *surface, FILE *f) {
  bool is_more_data = false;

  // Map makes surface writable by CPU for all implementations
  mfxStatus sts = surface->FrameInterface->Map(surface, MFX_MAP_WRITE);
  if (sts != MFX_ERR_NONE) {
    printf("mfxFrameSurfaceInterface->Map failed (%d)\n", sts);
    return sts;
  }

  sts = ReadRawFrame(surface, f);
  if (sts != MFX_ERR_NONE) {
    if (sts == MFX_ERR_MORE_DATA)
      is_more_data = true;
    else
      return sts;
  }

  // Unmap/release returns local device access for all implementations
  sts = surface->FrameInterface->Unmap(surface);
  if (sts != MFX_ERR_NONE) {
    printf("mfxFrameSurfaceInterface->Unmap failed (%d)\n", sts);
    return sts;
  }

  return (is_more_data == true) ? MFX_ERR_MORE_DATA : MFX_ERR_NONE;
}
#endif

// Write raw I420 frame to file
mfxStatus WriteRawFrame(mfxFrameSurface1 *surface, FILE *f) {
  mfxU16 w, h, i, pitch;
  mfxFrameInfo *info = &surface->Info;
  mfxFrameData *data = &surface->Data;

  w = info->Width;
  h = info->Height;

  // write the output to disk
  switch (info->FourCC) {
    case MFX_FOURCC_I420:
      // Y
      pitch = data->Pitch;
      for (i = 0; i < h; i++) {
        fwrite(data->Y + i * pitch, 1, w, f);
      }
      // U
      pitch /= 2;
      h /= 2;
      w /= 2;
      for (i = 0; i < h; i++) {
        fwrite(data->U + i * pitch, 1, w, f);
      }
      // V
      for (i = 0; i < h; i++) {
        fwrite(data->V + i * pitch, 1, w, f);
      }
      break;
    case MFX_FOURCC_NV12:
      // Y
      pitch = data->Pitch;
      for (i = 0; i < h; i++) {
        fwrite(data->Y + i * pitch, 1, w, f);
      }
      // UV
      h /= 2;
      for (i = 0; i < h; i++) {
        fwrite(data->UV + i * pitch, 1, w, f);
      }
      break;
    case MFX_FOURCC_RGB4:
      // Y
      pitch = data->Pitch;
      for (i = 0; i < h; i++) {
        fwrite(data->B + i * pitch, 1, pitch, f);
      }
      break;
    default:
      return MFX_ERR_UNSUPPORTED;
      break;
  }

  return MFX_ERR_NONE;
}

#if (MFX_VERSION >= 2000)
// Write raw frame to file
mfxStatus WriteRawFrame_InternalMem(mfxFrameSurface1 *surface, FILE *f) {
  mfxStatus sts = surface->FrameInterface->Map(surface, MFX_MAP_READ);
  if (sts != MFX_ERR_NONE) {
    printf("mfxFrameSurfaceInterface->Map failed (%d)\n", sts);
    return sts;
  }

  sts = WriteRawFrame(surface, f);
  if (sts != MFX_ERR_NONE) {
    printf("Error in WriteRawFrame\n");
    return sts;
  }

  sts = surface->FrameInterface->Unmap(surface);
  if (sts != MFX_ERR_NONE) {
    printf("mfxFrameSurfaceInterface->Unmap failed (%d)\n", sts);
    return sts;
  }

  sts = surface->FrameInterface->Release(surface);
  if (sts != MFX_ERR_NONE) {
    printf("mfxFrameSurfaceInterface->Release failed (%d)\n", sts);
    return sts;
  }

  return sts;
}
#endif

#endif  // EXAMPLES_UTIL_H_
